#!/usr/bin/env python3
"""
test_s172_phase4_coordinator.py — S172 Phase 4 coordinator acceptance harness

Gates numbered 1–36 exactly as `docs/S172_PHASE4_BRIEF.md` numbers them, PLUS gate
37 — the Team-Beta-required real-serve-path gate (added after Beta's serve-path
rejection). The harness therefore makes 37 `_check` calls total: the 36 brief
Phase-4 gates + gate 37; gate 23 additionally re-runs the Phase 0/1/2/3 harnesses
as a subprocess non-regression check. All are CPU-only, loopback, with stubbed
transfer + a stubbed Phase5Sink — no GPU, rig, or real SSH. Exit 0 = all green.
Exit 1 = a check failed (DO NOT COMMIT).

STAGED IMPLEMENTATION — gates land in stages (each stage green before the next):
  Stage 1 (ledger + state machine + reconciliation): gates 1, 2, 3, 4, 5, 6, 35.
  Stage 2 (identity/registration/fencing):            gates 17, 18, 19, 24, 29, 30.
  Stage 3 (staging pipeline + reservations):          gates 13, 14, 15, 16, 25, 26, 27, 31, 32, 36.
  Stage 4 (retry matrix + trial lifecycle + sink):    gates 7, 8, 9, 10, 11, 12, 28, 33, 34.
  Stage 5 (integration + coexistence + non-regression): gates 20, 21, 22, 23.

Gate list (brief §Test harness gates + rev-3/rev-4 additions):
   1. Macro-stripe partition + assign; macro MAY exceed one GPU cap;
      expected_substripes from advertised cap. (Blocker 7)
   2. Multiple sub-stripe results under one stripe → N shard rows by sub_index. (B1)
   3. Missing / duplicate / overlapping sub_index → stripe NOT done. (B1)
   4. Shard-level done conditions (StripeComplete + all sub_index + coverage +
      staged/verified + totals reconcile). (B1)
   5. Staging state: after StripeComplete → staging (not done); compute reclaim
      does NOT fire during staging; no duplicate reassign. (B5)
   6. StripeComplete before transfers finish → waits for staging verify. (B5)
  35. Full completion reconciliation (L8): all four invariants → done; break any
      one → NOT done.

Run:
    cd ~/distributed_prng_analysis
    source ~/venvs/torch/bin/activate
    PYTHONPATH=. python3 tests/test_s172_phase4_coordinator.py
"""
import dataclasses
import hashlib
import inspect
import json
import os
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
_SKIP = "\033[93mSKIP\033[0m"

_results = []


def _check(name, fn):
    try:
        fn()
        _results.append((name, True, None))
        print(f"  [{_PASS}] {name}")
    except Exception as e:
        _results.append((name, False, traceback.format_exc()))
        print(f"  [{_FAIL}] {name}: {e}")


from miner.range_miner_coordinator import (  # noqa: E402
    ST_CLAIMED,
    ST_DONE,
    ST_PENDING,
    ST_STAGING,
    SH_VERIFIED,
    SH_FAILED,
    ST_CANCELLED,
    CompletionCheck,
    CoordinatorConfig,
    MinerLedger,
    NodeConfig,
    Phase5Sink,
    RangeMinerCoordinator,
    StagingBackPressure,
    StagingError,
    StagingHashMismatch,
    StagingTimeout,
    TransferAdapter,
    TrialAborted,
    WorkerConnection,
    WorkerRecord,
    advertised_effective_cap,
    build_coordinator,
    evaluate_stripe_completion,
    event_id_for,
    expected_substripes_for,
    partition_macro_stripes,
    run_trial_miner,
    spool_path_within_root,
    workflow_stages_for,
)
import miner.range_miner_coordinator as COORD  # noqa: E402
from miner.range_miner_worker import (  # noqa: E402
    GpuInfo,
    MinerFramedSocket,
    ResidueResolver,
    RangeMinerWorker,
    SieveExecutor,
    VramCaps,
    build_substripe_payload_bytes,
    supported_variants,
)
from miner.range_miner_protocol import (  # noqa: E402
    MinerHeartbeatMessage,
    MinerShutdownMessage,
    RegisterMessage,
    StripeAssignMessage,
    StripeCompleteMessage,
    StripeErrorMessage,
    SubStripeResultMessage,
)

MACRO = 67_108_864
CAPS = {"amd": 2_000_000, "nvidia": 5_000_000,
        "amd_hybrid": 1_000_000, "nvidia_hybrid": 2_500_000}


def _worker(wid="hostA:gpu0", backend="cuda", caps=None):
    return WorkerRecord(worker_id=wid, backend=backend,
                        seed_caps=dict(caps or CAPS), hostname=wid.split(":")[0])


def _coord(tmp, **cfg):
    ledger = MinerLedger(os.path.join(tmp, "ledger.db"))
    return RangeMinerCoordinator(CoordinatorConfig(**cfg), ledger)


def _record_and_verify_all(coord, run_id, stripe_id, worker_id, attempt,
                           seed_start, seed_count, n, survivors_each=1, now=100.0):
    """Record n contiguous sub-stripe results tiling the stripe, then mark each
    verified. Returns (per_sub_seed_count list, total_survivors)."""
    base = seed_count // n
    counts = [base] * n
    counts[-1] += seed_count - base * n            # last absorbs remainder
    cursor = seed_start
    for i in range(n):
        ok = coord.ledger.record_substripe_result(
            run_id, stripe_id, attempt, i, worker_id, cursor, counts[i],
            survivors_each, remote_spool_path=f"/spool/{stripe_id}/{i}.json",
            size_bytes=100, sha256="abc", now=now)
        assert ok, f"record sub {i} failed"
        cursor += counts[i]
    for i in range(n):
        coord.ledger.mark_shard_verified(
            run_id, stripe_id, attempt, i,
            local_staged_path=f"/staged/{stripe_id}/{i}.json", now=now)
    return counts, survivors_each * n


# ---------------------------------------------------------------------------
# GATE 1 — macro-stripe partition + assign
# ---------------------------------------------------------------------------
def gate1_macro_partition_assign():
    # pure partitioner: contiguous, no gap/overlap, tail smaller
    parts = partition_macro_stripes(2 * MACRO + 500, MACRO, base_start=0)
    assert [(p[1], p[2]) for p in parts] == [(0, MACRO), (MACRO, MACRO), (2 * MACRO, 500)]
    cursor = 0
    for _, s, c in parts:
        assert s == cursor and c > 0
        cursor += c
    assert cursor == 2 * MACRO + 500
    assert partition_macro_stripes(0, MACRO) == []
    try:
        partition_macro_stripes(10, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("macro_size<=0 must raise")

    # expected_substripes from ADVERTISED cap; macro (67M) MAY exceed a GPU cap.
    assert expected_substripes_for(MACRO, 5_000_000) == 14        # constant nvidia
    assert expected_substripes_for(MACRO, 2_500_000) == 27        # hybrid nvidia
    assert advertised_effective_cap("cuda", "java_lcg", CAPS) == 5_000_000
    assert advertised_effective_cap("cuda", "java_lcg_hybrid", CAPS) == 2_500_000
    assert advertised_effective_cap("rocm", "lcg32_hybrid_reverse", CAPS) == 1_000_000

    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp, miner_stripe_size=MACRO)
        w = _worker()
        a = coord.assign_stripes("run1", "java_lcg", 1, 2 * MACRO + 500, [w], now=100.0)
        assert len(a) == 3
        # coverage of the assigned macro-stripes is exact
        cur = 0
        for rec in a:
            assert rec["seed_start"] == cur and rec["claimed"] is True
            assert rec["worker_id"] == w.worker_id
            cur += rec["seed_count"]
        assert cur == 2 * MACRO + 500
        # expected_substripes from the worker's advertised constant cap (5M)
        assert [r["expected_substripes"] for r in a] == [14, 14, 1]
        for rec in a:
            st = coord.ledger.get_stripe("run1", rec["stripe_id"])
            assert st["state"] == ST_CLAIMED and st["current_attempt"] == 0

        # hybrid family selects the tighter advertised cap -> more sub-stripes
        h = coord.assign_stripes("runH", "java_lcg_hybrid", 3, MACRO, [w], now=100.0)
        assert h[0]["expected_substripes"] == 27
        assert h[0]["effective_cap"] == 2_500_000


# ---------------------------------------------------------------------------
# GATE 2 — multiple sub-stripe results under one stripe (shard-level ledger)
# ---------------------------------------------------------------------------
def gate2_multiple_substripe_results():
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp, miner_stripe_size=1000)
        w = _worker(caps={**CAPS, "nvidia": 10})   # cap 10 -> 3 sub-stripes over 30
        a = coord.assign_stripes("r", "java_lcg", 1, 30, [w], now=100.0)
        assert len(a) == 1 and a[0]["expected_substripes"] == 3
        sid = a[0]["stripe_id"]
        # three distinct sub-stripe results
        for i, (s, c, sv) in enumerate([(0, 10, 2), (10, 10, 5), (20, 10, 0)]):
            assert coord.ledger.record_substripe_result(
                "r", sid, 0, i, w.worker_id, s, c, sv,
                remote_spool_path=f"/spool/{i}", size_bytes=10, sha256="h", now=100.0)
        rows = coord.ledger.get_shards("r", sid, 0)
        assert len(rows) == 3
        assert [row["sub_index"] for row in rows] == [0, 1, 2]
        assert [row["survivor_count"] for row in rows] == [2, 5, 0]
        assert [row["seed_start"] for row in rows] == [0, 10, 20]
        # a one-row-per-stripe table would have overwritten -> assert it did not
        assert sum(row["seed_count"] for row in rows) == 30


# ---------------------------------------------------------------------------
# GATE 3 — missing / duplicate / overlapping sub_index → NOT done
# ---------------------------------------------------------------------------
def gate3_bad_sub_index_not_done():
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp, miner_stripe_size=1000)
        w = _worker(caps={**CAPS, "nvidia": 10})

        # MISSING: only 2 of 3 sub-stripes recorded + verified
        am = coord.assign_stripes("miss", "java_lcg", 1, 30, [w], now=100.0)[0]
        sid = am["stripe_id"]
        for i, (s, c) in enumerate([(0, 10), (10, 10)]):
            coord.ledger.record_substripe_result("miss", sid, 0, i, w.worker_id, s, c, 0,
                                                  size_bytes=1, sha256="h", now=100.0)
            coord.ledger.mark_shard_verified("miss", sid, 0, i, now=100.0)
        coord.ledger.record_stripe_complete("miss", sid, 0, w.worker_id, 3, 0)
        chk = coord.finalize_stripe("miss", sid)
        assert not chk.is_complete
        assert coord.ledger.get_stripe("miss", sid)["state"] != ST_DONE

        # DUPLICATE: sub_index 1 recorded twice — the second is REJECTED, not overwritten
        ad = coord.assign_stripes("dup", "java_lcg", 1, 30, [w], now=100.0)[0]
        sid = ad["stripe_id"]
        assert coord.ledger.record_substripe_result("dup", sid, 0, 1, w.worker_id, 10, 10, 0,
                                                     size_bytes=1, sha256="h", now=100.0) is True
        assert coord.ledger.record_substripe_result("dup", sid, 0, 1, w.worker_id, 10, 10, 99,
                                                     size_bytes=1, sha256="h", now=100.0) is False
        rows = coord.ledger.get_shards("dup", sid, 0)
        assert len(rows) == 1 and rows[0]["survivor_count"] == 0   # first kept, no overwrite
        chk = coord.finalize_stripe("dup", sid)
        assert not chk.is_complete

        # OVERLAPPING seed ranges across distinct sub_index -> coverage rejects
        ao = coord.assign_stripes("ovl", "java_lcg", 1, 30, [w], now=100.0)[0]
        sid = ao["stripe_id"]
        for i, (s, c) in enumerate([(0, 10), (5, 10), (20, 10)]):   # 0-10 and 5-15 overlap
            coord.ledger.record_substripe_result("ovl", sid, 0, i, w.worker_id, s, c, 0,
                                                  size_bytes=1, sha256="h", now=100.0)
            coord.ledger.mark_shard_verified("ovl", sid, 0, i, now=100.0)
        coord.ledger.record_stripe_complete("ovl", sid, 0, w.worker_id, 3, 0)
        chk = coord.finalize_stripe("ovl", sid)
        assert chk.coverage_ok is False and not chk.is_complete
        assert coord.ledger.get_stripe("ovl", sid)["state"] == ST_STAGING


# ---------------------------------------------------------------------------
# GATE 4 — shard-level done conditions
# ---------------------------------------------------------------------------
def gate4_done_conditions():
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp, miner_stripe_size=1000)
        coord.ledger.set_trial_context("r", _ctx())
        coord.ledger.set_trial_context("r2", _ctx())
        w = _worker(caps={**CAPS, "nvidia": 10})
        a = coord.assign_stripes("r", "java_lcg", 1, 30, [w], now=100.0)[0]
        sid = a["stripe_id"]

        # record all 3 sub-stripes and verify them
        _record_and_verify_all(coord, "r", sid, w.worker_id, 0, 0, 30, 3,
                               survivors_each=2, now=100.0)

        # NOT done before StripeComplete (no reconciliation input yet)
        assert not coord.finalize_stripe("r", sid).is_complete
        assert coord.ledger.get_stripe("r", sid)["state"] == ST_CLAIMED

        # StripeComplete with matching totals -> staging, then finalize -> done
        assert coord.ledger.record_stripe_complete("r", sid, 0, w.worker_id, 3, 6)
        assert coord.ledger.get_stripe("r", sid)["state"] == ST_STAGING
        chk = coord.finalize_stripe("r", sid)
        assert chk.is_complete, chk.reasons
        assert coord.ledger.get_stripe("r", sid)["state"] == ST_DONE

        # a stripe missing verification never reaches done
        a2 = coord.assign_stripes("r2", "java_lcg", 1, 30, [w], now=100.0)[0]
        sid2 = a2["stripe_id"]
        for i, (s, c) in enumerate([(0, 10), (10, 10), (20, 10)]):
            coord.ledger.record_substripe_result("r2", sid2, 0, i, w.worker_id, s, c, 2,
                                                  size_bytes=1, sha256="h", now=100.0)
        coord.ledger.mark_shard_verified("r2", sid2, 0, 0, now=100.0)   # only ONE verified
        coord.ledger.record_stripe_complete("r2", sid2, 0, w.worker_id, 3, 6)
        chk2 = coord.finalize_stripe("r2", sid2)
        assert chk2.reconciled and not chk2.all_verified and not chk2.is_complete
        assert coord.ledger.get_stripe("r2", sid2)["state"] == ST_STAGING


# ---------------------------------------------------------------------------
# GATE 5 — staging state; compute reclaim skips staging; no duplicate reassign
# ---------------------------------------------------------------------------
def gate5_staging_state_reclaim():
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp, miner_stripe_size=1000, compute_lease_timeout=50.0)
        w = _worker(caps={**CAPS, "nvidia": 10})

        # stripe A: completes -> staging (lease cleared)
        a = coord.assign_stripes("run", "java_lcg", 1, 30, [w], now=100.0)[0]
        sidA = a["stripe_id"]
        _record_and_verify_all(coord, "run", sidA, w.worker_id, 0, 0, 30, 3, now=100.0)
        coord.ledger.record_stripe_complete("run", sidA, 0, w.worker_id, 3, 3)
        assert coord.ledger.get_stripe("run", sidA)["state"] == ST_STAGING
        assert coord.ledger.get_stripe("run", sidA)["lease_expires_at"] is None

        # stripe B: stays claimed, lease will expire
        coord.ledger.add_stripe("run", "run_sB", 100, 30, "java_lcg", 1, now=100.0)
        coord.ledger.claim_stripe("run", "run_sB", w.worker_id, 0, 3,
                                  lease_expires_at=150.0)

        # advance the clock well past both original leases and reclaim
        reclaimed = coord.ledger.reclaim_expired_leases("run", now=1000.0)
        reclaimed_ids = {r["stripe_id"] for r in reclaimed}
        # ONLY the claimed stripe B is reclaimed; the STAGING stripe A is untouched
        assert "run_sB" in reclaimed_ids
        assert sidA not in reclaimed_ids, "staging stripe must NOT be compute-reclaimed"
        assert coord.ledger.get_stripe("run", sidA)["state"] == ST_STAGING
        assert coord.ledger.get_stripe("run", "run_sB")["state"] == ST_PENDING
        # no duplicate reassignment of the staging stripe (still exactly one row, staging)
        assert len(coord.ledger.stripes_by_state("run", ST_STAGING)) == 1


# ---------------------------------------------------------------------------
# GATE 6 — StripeComplete before transfers finish → waits for staging verify
# ---------------------------------------------------------------------------
def gate6_wait_for_staging():
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp, miner_stripe_size=1000)
        coord.ledger.set_trial_context("run", _ctx())
        w = _worker(caps={**CAPS, "nvidia": 10})
        a = coord.assign_stripes("run", "java_lcg", 1, 30, [w], now=100.0)[0]
        sid = a["stripe_id"]
        # results recorded but NOT yet staged/verified (transfers still running)
        for i, (s, c) in enumerate([(0, 10), (10, 10), (20, 10)]):
            coord.ledger.record_substripe_result("run", sid, 0, i, w.worker_id, s, c, 1,
                                                  size_bytes=1, sha256="h", now=100.0)
        coord.ledger.record_stripe_complete("run", sid, 0, w.worker_id, 3, 3)
        assert coord.ledger.get_stripe("run", sid)["state"] == ST_STAGING

        # StripeComplete seen + reconciled, but staging not verified -> NOT done
        chk = coord.finalize_stripe("run", sid)
        assert chk.reconciled and not chk.all_verified and not chk.is_complete
        assert coord.ledger.get_stripe("run", sid)["state"] == ST_STAGING

        # transfers finish + verify -> now done
        for i in range(3):
            coord.ledger.mark_shard_verified("run", sid, 0, i, now=100.0)
        chk = coord.finalize_stripe("run", sid)
        assert chk.is_complete
        assert coord.ledger.get_stripe("run", sid)["state"] == ST_DONE


# ---------------------------------------------------------------------------
# GATE 35 — full completion reconciliation (L8); break any one invariant → NOT done
# ---------------------------------------------------------------------------
def _good_stripe_and_shards():
    stripe = {
        "seed_start": 0, "seed_count": 30, "expected_substripes": 3,
        "stripe_complete_seen": True, "substripes_done": 3, "survivors_total": 6,
    }
    shards = [
        {"sub_index": 0, "seed_start": 0, "seed_count": 10, "survivor_count": 2,
         "staging_status": SH_VERIFIED},
        {"sub_index": 1, "seed_start": 10, "seed_count": 10, "survivor_count": 3,
         "staging_status": SH_VERIFIED},
        {"sub_index": 2, "seed_start": 20, "seed_count": 10, "survivor_count": 1,
         "staging_status": SH_VERIFIED},
    ]
    return stripe, shards


def gate35_full_reconciliation():
    stripe, shards = _good_stripe_and_shards()
    ok = evaluate_stripe_completion(stripe, shards)
    assert ok.is_complete and ok.reconciled and ok.all_verified, ok.reasons

    # break substripes_done (short a sub_index / expected mismatch)
    s, sh = _good_stripe_and_shards()
    bad = evaluate_stripe_completion(s, sh[:-1])                       # drop sub_index 2
    assert not bad.substripes_match and not bad.is_complete

    # seed_count sum off
    s, sh = _good_stripe_and_shards()
    sh[2]["seed_count"] = 9                                            # sum 29 != 30
    bad = evaluate_stripe_completion(s, sh)
    assert not bad.seed_sum_match and not bad.is_complete

    # survivor sum off
    s, sh = _good_stripe_and_shards()
    sh[0]["survivor_count"] = 99                                       # sum 103 != 6
    bad = evaluate_stripe_completion(s, sh)
    assert not bad.survivor_sum_match and not bad.is_complete

    # coverage gap
    s, sh = _good_stripe_and_shards()
    sh[1]["seed_start"] = 11                                           # gap at 10..11
    bad = evaluate_stripe_completion(s, sh)
    assert not bad.coverage_ok and not bad.is_complete

    # not every shard verified
    s, sh = _good_stripe_and_shards()
    sh[1]["staging_status"] = "pending"
    bad = evaluate_stripe_completion(s, sh)
    assert not bad.all_verified and not bad.is_complete
    # ...but the four accounting invariants still hold (reconciled True)
    assert bad.reconciled


SPOOL_ROOT = "/var/spool/miner"
VARIANTS = ["java_lcg", "java_lcg_reverse", "java_lcg_hybrid", "java_lcg_hybrid_reverse"]


def _register(coord, wid="hostA:gpu0", backend="cuda", caps=None,
              variants=None, spool_root=SPOOL_ROOT, now=100.0):
    node = NodeConfig(hostname=wid.split(":")[0], spool_root=spool_root,
                      ssh_address="10.0.0.9", ssh_user="michael")
    return coord.register_worker(
        worker_id=wid, hostname=wid.split(":")[0], backend=backend,
        capabilities={"seed_caps": dict(caps or CAPS),
                      "supported_variants": list(variants or VARIANTS)},
        node_config=node, now=now)


# ---------------------------------------------------------------------------
# GATE 17 — spool path restricted to the worker's configured spool root
# ---------------------------------------------------------------------------
def gate17_spool_root_restriction():
    # pure helper: normalized, absolute, strictly-under-root; `..` collapses out
    assert spool_path_within_root(SPOOL_ROOT, "/var/spool/miner/r/2.json") is True
    assert spool_path_within_root(SPOOL_ROOT, "/var/spool/miner") is False        # root itself
    assert spool_path_within_root(SPOOL_ROOT, "/etc/passwd") is False
    assert spool_path_within_root(SPOOL_ROOT, "/var/spool/miner/../../etc/x") is False
    assert spool_path_within_root(SPOOL_ROOT, "/var/spool/miner/a/../b.json") is True
    assert spool_path_within_root(SPOOL_ROOT, "relative/x") is False              # not absolute
    assert spool_path_within_root("", "/var/spool/miner/x") is False

    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp)
        conn = _register(coord)
        assert coord.validate_spool_path(conn, "/var/spool/miner/run_s0/3.json") is True
        # a manifest whose path escapes the worker's configured spool root is rejected
        assert coord.validate_spool_path(conn, "/var/spool/miner/../evil/3.json") is False
        assert coord.validate_spool_path(conn, "/home/michael/secret") is False


# ---------------------------------------------------------------------------
# GATE 18 — connection-bound identity (Decision A)
# ---------------------------------------------------------------------------
def gate18_connection_bound_identity():
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp, miner_stripe_size=1000)
        connA = _register(coord, "hostA:gpu0")
        a = coord.assign_stripes("run", "java_lcg", 1, 30, [connA], now=100.0)[0]
        sid = a["stripe_id"]
        assert a["claimed"] and connA.assignment_attempts[sid] == 0

        ok, _ = coord.accept_stripe_message(connA, "run", sid, "hostA:gpu0",
                                            (ST_CLAIMED, ST_STAGING))
        assert ok is True

        # a message whose worker_id does not match the bound connection is rejected
        ok, reason = coord.accept_stripe_message(connA, "run", sid, "hostB:gpu0",
                                                 (ST_CLAIMED, ST_STAGING))
        assert ok is False and "bound connection" in reason


# ---------------------------------------------------------------------------
# GATE 19 — cap / supported-variant mismatch at registration
# ---------------------------------------------------------------------------
def gate19_cap_or_variant_mismatch():
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp, miner_stripe_size=1000)

        # (a) caps inconsistent with central config -> quarantined
        bad_caps = {**CAPS, "nvidia_hybrid": 9_999_999}
        connBad = _register(coord, "hostBad:gpu0", caps=bad_caps)
        assert connBad.quarantined is True and "nvidia_hybrid" in connBad.quarantine_reason
        assert coord.can_assign_variant(connBad, "java_lcg") is False   # ineligible

        # (b) consistent caps but a variant it cannot support -> not assignable
        connOK = _register(coord, "hostOK:gpu0",
                           variants=["java_lcg", "java_lcg_hybrid"])
        assert connOK.quarantined is False
        assert coord.can_assign_variant(connOK, "java_lcg") is True
        assert coord.can_assign_variant(connOK, "mt19937") is False

        # assignment of an unsupported variant is refused (stripe stays pending)
        res = coord.assign_stripes("run", "mt19937", 1, 30, [connOK], now=100.0)[0]
        assert res["claimed"] is False and "cannot serve" in res["refused_reason"]
        assert coord.ledger.get_stripe("run", res["stripe_id"])["state"] == ST_PENDING


# ---------------------------------------------------------------------------
# GATE 24 — stale-attempt fencing (L1)
# ---------------------------------------------------------------------------
def gate24_stale_attempt_fencing():
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp, compute_lease_timeout=50.0)
        connA = _register(coord, "hostA:gpu0")
        connB = _register(coord, "hostB:gpu0")

        # attempt-0 assigned to worker A (lease 100..150), one result recorded
        sid = "run_sX"
        coord.ledger.add_stripe("run", sid, 0, 30, "java_lcg", 1, now=100.0)
        assert coord.ledger.claim_stripe("run", sid, "hostA:gpu0", 0, 3,
                                         lease_expires_at=150.0)
        connA.record_assignment(sid, 0)
        ok, _ = coord.accept_stripe_message(connA, "run", sid, "hostA:gpu0",
                                            (ST_CLAIMED,))
        assert ok
        coord.ledger.record_substripe_result("run", sid, 0, 0, "hostA:gpu0", 0, 10, 1,
                                             size_bytes=1, sha256="h", now=110.0)

        # lease expires -> reclaim (fences attempt 0, bumps staging_generation)
        reclaimed = coord.ledger.reclaim_expired_leases("run", now=1000.0)
        assert {r["stripe_id"] for r in reclaimed} == {sid}
        gen_after = coord.ledger.get_stripe("run", sid)["staging_generation"]
        assert gen_after == 1

        # attempt 1 assigned to worker B
        assert coord.ledger.claim_stripe("run", sid, "hostB:gpu0", 1, 3,
                                         lease_expires_at=1100.0)
        connB.record_assignment(sid, 1)

        stripe_before = coord.ledger.get_stripe("run", sid)
        shards_before = coord.ledger.get_shards("run", sid, 1)

        # a delayed result/complete from worker A now arrives -> REJECTED
        ok, reason = coord.accept_stripe_message(connA, "run", sid, "hostA:gpu0",
                                                 (ST_CLAIMED, ST_STAGING))
        assert ok is False and "stale" in reason

        # attempt-1 ledger is unchanged; no attempt-1 shard from the stale worker
        stripe_after = coord.ledger.get_stripe("run", sid)
        assert stripe_after["claimed_by"] == "hostB:gpu0"
        assert stripe_after["current_attempt"] == 1
        assert stripe_after == stripe_before
        assert coord.ledger.get_shards("run", sid, 1) == shards_before == []
        # attempt-0's shard is still keyed under attempt 0 (never migrated)
        assert len(coord.ledger.get_shards("run", sid, 0)) == 1


# ---------------------------------------------------------------------------
# GATE 29 — four-cap validation + quarantine (L4)
# ---------------------------------------------------------------------------
def gate29_four_cap_validation():
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp)

        # all four caps consistent -> eligible, durably recorded
        good = _register(coord, "good:gpu0")
        assert good.quarantined is False
        assert coord.ledger.get_worker("good:gpu0")["status"] == "eligible"

        # missing hybrid cap -> quarantine
        miss = _register(coord, "miss:gpu0",
                         caps={"amd": 2_000_000, "nvidia": 5_000_000,
                               "amd_hybrid": 1_000_000})   # no nvidia_hybrid
        assert miss.quarantined and "nvidia_hybrid" in miss.quarantine_reason

        # zero / non-positive cap -> quarantine
        zero = _register(coord, "zero:gpu0", caps={**CAPS, "amd_hybrid": 0})
        assert zero.quarantined and "amd_hybrid" in zero.quarantine_reason

        # mismatched value -> quarantine
        mism = _register(coord, "mism:gpu0", caps={**CAPS, "nvidia": 4_000_000})
        assert mism.quarantined and "nvidia" in mism.quarantine_reason

        # all quarantines are durably visible (registered-but-ineligible)
        for wid in ("miss:gpu0", "zero:gpu0", "mism:gpu0"):
            assert coord.ledger.get_worker(wid)["status"] == "quarantined"
            assert coord.ledger.get_worker(wid)["quarantine_reason"]


# ---------------------------------------------------------------------------
# GATE 30 — staging resources injectable config (L4)
# ---------------------------------------------------------------------------
def gate30_staging_resources_configurable():
    # run_trial_miner exposes the L4 params (not buried constants)
    params = inspect.signature(run_trial_miner).parameters
    for p in ("seed_cap_nvidia_hybrid", "seed_cap_amd_hybrid",
              "staging_high_water_bytes", "staging_high_water_files",
              "staging_dir", "compute_lease_timeout", "staging_timeout"):
        assert p in params, f"run_trial_miner missing L4 param {p}"

    # CoordinatorConfig accepts + honors them (behavioral, not just present)
    with tempfile.TemporaryDirectory() as tmp:
        cfg = CoordinatorConfig(
            miner_stripe_size=1000,
            staging_high_water_bytes=123, staging_high_water_files=7,
            staging_dir="/mnt/staging", compute_lease_timeout=42.0,
            staging_timeout=99.0)
        coord = RangeMinerCoordinator(cfg, MinerLedger(os.path.join(tmp, "l.db")))
        assert coord.config.staging_high_water_bytes == 123
        assert coord.config.staging_high_water_files == 7
        assert coord.config.staging_dir == "/mnt/staging"
        assert coord.config.staging_timeout == 99.0
        # compute_lease_timeout drives the lease the assignment records
        conn = _register(coord, "hostA:gpu0")
        a = coord.assign_stripes("run", "java_lcg", 1, 30, [conn], now=100.0)[0]
        st = coord.ledger.get_stripe("run", a["stripe_id"])
        assert st["lease_expires_at"] == 142.0   # now(100) + compute_lease_timeout(42)


# ---------------------------------------------------------------------------
# Stage 3 — staging pipeline + reservations (gates 13-16, 25-27, 31, 32, 36)
# ---------------------------------------------------------------------------
class _StubTransfer(TransferAdapter):
    """Stubbed Decision-B adapter. fetch_remote writes preset bytes to the local
    temp; delete_remote records invocations (and optionally the shard's staging
    status AT the moment of deletion, to prove delete-after-verify ordering)."""

    def __init__(self, payloads=None, fail_fetch=False, fail_delete=False,
                 status_probe=None, fetch_gate=None, fetch_delay=0.0,
                 fetch_fail_after_delay=False):
        self.payloads = payloads or {}
        self.fail_fetch = fail_fetch
        self.fail_delete = fail_delete
        self.status_probe = status_probe
        # Defect 4 knobs: fetch_gate blocks until set (concurrency probe);
        # fetch_delay + fetch_fail_after_delay simulate a slow fetch that never
        # delivers (staging-timeout probe — it raises WITHOUT writing the temp).
        self.fetch_gate = fetch_gate
        self.fetch_delay = fetch_delay
        self.fetch_fail_after_delay = fetch_fail_after_delay
        self.fetch_calls = []
        self.delete_calls = []
        self.status_seen_at_delete = []

    def fetch_remote(self, node_config, remote_path, local_temp_path):
        self.fetch_calls.append(remote_path)
        if self.fetch_gate is not None:
            self.fetch_gate.wait(timeout=10)
        if self.fetch_delay:
            time.sleep(self.fetch_delay)
        if self.fetch_fail_after_delay:
            raise IOError("simulated slow fetch that never delivers")
        if self.fail_fetch:
            raise IOError("simulated fetch failure")
        with open(local_temp_path, "wb") as f:
            f.write(self.payloads.get(remote_path, b""))

    def delete_remote(self, node_config, remote_path):
        self.delete_calls.append(remote_path)
        if self.status_probe is not None:
            self.status_seen_at_delete.append(self.status_probe())
        if self.fail_delete:
            raise IOError("simulated delete failure")


def _coord_staging(tmp, transfer=None, sink=None, dbname="l.db", **cfg):
    cfg.setdefault("staging_dir", os.path.join(tmp, "staging"))
    ledger = MinerLedger(os.path.join(tmp, dbname))
    return RangeMinerCoordinator(CoordinatorConfig(**cfg), ledger,
                                 transfer=transfer, phase5_sink=sink)


class _StubSink(Phase5Sink):
    """Stubbed Phase-5 interface. abort_trial is SYNCHRONOUS (Option A): it may
    probe state (e.g. that a staged file still exists during the call) and may be
    made to raise to exercise the L7 retain-on-failure path."""

    def __init__(self, abort_probe=None, abort_raises=False):
        self.published = []
        self.commits = []
        self.aborts = []
        self.abort_probe = abort_probe
        self.abort_raises = abort_raises
        # Defect 5: dedup commit/abort by immutable event_id + record the thread
        # the abort discharge ran on (must be the off-dispatch cleanup executor).
        self.commit_event_ids = set()
        self.abort_event_ids = set()
        self.abort_threads = []

    def publish_shard(self, manifest):
        self.published.append(manifest)

    def commit_trial(self, event):
        self.commits.append(event)
        self.commit_event_ids.add(event.get("event_id"))

    def abort_trial(self, event):
        self.aborts.append(event)
        self.abort_event_ids.add(event.get("event_id"))
        self.abort_threads.append(threading.current_thread().name)
        if self.abort_probe is not None:
            self.abort_probe()
        if self.abort_raises:
            raise IOError("simulated Phase-5 abort failure")


def _ctx(**over):
    """D0 REV3 (Blocker 1): a valid trial-global context so publish_attempt can
    reconstruct a complete trial_metadata. Blocker 1 now FAILS CLOSED when the
    durable trial_context row is absent, so every gate that PUBLISHES must persist
    this first — production always does via serve_trial/build_trial_context_from_serve."""
    c = dict(trial_number=7, window_size=5, offset=2,
             sessions=["midday", "evening"], skip_min=1, skip_max=9,
             prng_base="java_lcg", forward_threshold=0.40, reverse_threshold=0.45,
             dataset_sha256="d" * 64, residue_sha256="r" * 64)
    c.update(over)
    return c


def _stage_complete_inline(coord, sid, conn, survivors, now=100.0):
    """Stage a single sub-stripe covering the whole 30-seed stripe, record
    StripeComplete, and finalize -> attempt published + stripe done."""
    coord.ledger.set_trial_context("run", _ctx())
    pb, size, sha = _canon(sid, 0, 0, 30, survivors)
    coord.ledger.record_substripe_result(
        "run", sid, 0, 0, conn.worker_id, 0, 30, len(survivors),
        remote_spool_path=None, size_bytes=size, sha256=sha, now=now)
    res = coord.stage_inline_shard("run", sid, 0, 0, 0, 30, survivors, size, sha, now=now)
    coord.ledger.record_stripe_complete("run", sid, 0, conn.worker_id, 1, len(survivors))
    coord.finalize_stripe("run", sid, now=now)
    return res, size


def _canon(stripe_id, sub_index, seed_start, seed_count, survivors):
    _, pb = build_substripe_payload_bytes(stripe_id, sub_index, seed_start,
                                          seed_count, survivors)
    return pb, len(pb), hashlib.sha256(pb).hexdigest()


def _assign_one(coord, run="run", family="java_lcg", total=30, now=100.0):
    conn = _register(coord)
    a = coord.assign_stripes(run, family, 1, total, [conn], now=now)[0]
    return conn, a["stripe_id"]


# ---------------------------------------------------------------------------
# GATE 13 — inline result normalized to a Zeus-local file manifest (B4)
# ---------------------------------------------------------------------------
def gate13_inline_normalized():
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord_staging(tmp, miner_stripe_size=1000)
        coord.ledger.set_trial_context("run", _ctx())
        conn, sid = _assign_one(coord)   # total 30, default caps -> expected_substripes 1
        survivors = [[0, 0.9, None, [1]], [5, 0.8, None, [2, 3]]]
        pb, size, sha = _canon(sid, 0, 0, 30, survivors)
        coord.ledger.record_substripe_result(
            "run", sid, 0, 0, conn.worker_id, 0, 30, 2,
            remote_spool_path=None, size_bytes=size, sha256=sha, now=100.0)
        res = coord.stage_inline_shard("run", sid, 0, 0, 0, 30, survivors,
                                       size, sha, now=100.0)
        assert res["status"] == "verified"
        staged = res["staged_path"]
        assert os.path.isfile(staged)
        raw = open(staged, "rb").read()
        # byte-identical to the worker's canonical s172_substripe_v1 serialization
        assert raw == pb, "inline normalization must match build_substripe_payload_bytes"
        assert len(raw) == size and hashlib.sha256(raw).hexdigest() == sha
        obj = json.loads(raw.decode("utf-8"))
        assert obj["schema_version"] == "s172_substripe_v1"
        assert obj["stripe_id"] == sid and obj["sub_index"] == 0
        # Blocker 2: NOTHING is published to Phase 5 until the whole attempt is done
        assert coord.enqueued == []
        assert coord.ledger.get_shard("run", sid, 0, 0)["staging_status"] == SH_VERIFIED

        # complete the (single-sub) stripe -> attempt published as the uniform manifest
        coord.ledger.record_stripe_complete("run", sid, 0, conn.worker_id, 1, 2)
        coord.finalize_stripe("run", sid, now=100.0)
        assert coord.ledger.get_stripe("run", sid)["state"] == ST_DONE
        m = coord.enqueued[-1]
        # SAME uniform path-manifest shape as remote spools, with the L6 event_id
        for k in ("event_id", "local_spool_path", "expected_size", "expected_sha256",
                  "stripe_id", "attempt", "sub_index", "trial_metadata"):
            assert k in m, k
        assert m["local_spool_path"] == staged and m["expected_sha256"] == sha
        assert coord.ledger.get_shard("run", sid, 0, 0)["phase5_status"] == "enqueued"


# ---------------------------------------------------------------------------
# GATE 14 — remote staging happy path; delete_remote only after verify (Dec. B)
# ---------------------------------------------------------------------------
def gate14_remote_happy_path():
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord_staging(tmp, miner_stripe_size=1000)
        conn, sid = _assign_one(coord)
        survivors = [[1, 0.9, 0, [1, 2, 3]]]
        pb, size, sha = _canon(sid, 0, 0, 10, survivors)
        remote = f"/var/spool/miner/{sid}/0.json"
        stub = _StubTransfer(
            payloads={remote: pb},
            status_probe=lambda: coord.ledger.get_shard("run", sid, 0, 0)["staging_status"])
        coord.transfer = stub
        coord.ledger.record_substripe_result(
            "run", sid, 0, 0, conn.worker_id, 0, 10, 1,
            remote_spool_path=remote, size_bytes=size, sha256=sha, now=100.0)
        res = coord.stage_remote_shard(conn, "run", sid, 0, 0, remote, size, sha, now=100.0)
        assert res["status"] == "verified"
        assert os.path.isfile(res["staged_path"])
        assert open(res["staged_path"], "rb").read() == pb
        assert stub.fetch_calls == [remote]
        # delete_remote invoked exactly once, and ONLY after the shard was verified
        assert stub.delete_calls == [remote]
        assert stub.status_seen_at_delete == [SH_VERIFIED]
        sh = coord.ledger.get_shard("run", sid, 0, 0)
        assert sh["staging_status"] == SH_VERIFIED and sh["remote_delete_status"] == "deleted"
        assert res["manifest"]["local_spool_path"] == res["staged_path"]


# ---------------------------------------------------------------------------
# GATE 15 — hash mismatch → failed sub-stripe; delete_remote NOT invoked (§15)
# ---------------------------------------------------------------------------
def gate15_hash_mismatch():
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord_staging(tmp, miner_stripe_size=1000)
        conn, sid = _assign_one(coord)
        survivors = [[1, 0.9, 0, [1]]]
        pb, size, sha = _canon(sid, 0, 0, 10, survivors)
        remote = f"/var/spool/miner/{sid}/0.json"
        stub = _StubTransfer(payloads={remote: pb + b"CORRUPT"})   # bytes != advertised sha
        coord.transfer = stub
        coord.ledger.record_substripe_result(
            "run", sid, 0, 0, conn.worker_id, 0, 10, 1,
            remote_spool_path=remote, size_bytes=size, sha256=sha, now=100.0)
        try:
            coord.stage_remote_shard(conn, "run", sid, 0, 0, remote, size, sha, now=100.0)
        except StagingHashMismatch:
            pass
        else:
            raise AssertionError("hash mismatch must raise StagingHashMismatch")
        assert stub.delete_calls == [], "delete_remote must NOT run on a mismatch"
        assert coord.ledger.get_shard("run", sid, 0, 0)["staging_status"] == SH_FAILED
        # reservation released, temp removed (no leak, nothing staged)
        assert coord.reserved_bytes() == 0 and coord.reserved_files() == 0
        assert os.listdir(coord.config.staging_dir) == []


# ---------------------------------------------------------------------------
# GATE 16 — byte reservation / high-water mark (§15)
# ---------------------------------------------------------------------------
def gate16_high_water_mark():
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord_staging(tmp, staging_high_water_bytes=100, staging_high_water_files=10)
        r1 = coord.reserve_capacity("run", "s", 0, 0, 0, 60, now=1.0)
        r2 = coord.reserve_capacity("run", "s", 0, 1, 0, 30, now=1.0)
        assert r1 is not None and r2 is not None and coord.reserved_bytes() == 90
        # a transfer that would exceed the mark is back-pressured; staged never exceeds
        assert coord.reserve_capacity("run", "s", 0, 2, 0, 20, now=1.0) is None
        assert coord.reserved_bytes() == 90
        # freeing capacity lets the next transfer fit
        coord.ledger.release_reservation(r1, now=2.0)
        assert coord.reserved_bytes() == 30
        assert coord.reserve_capacity("run", "s", 0, 2, 0, 20, now=2.0) is not None
        assert coord.reserved_bytes() == 50

    # file-count mark is enforced independently
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord_staging(tmp, staging_high_water_bytes=10 ** 9,
                               staging_high_water_files=2)
        assert coord.reserve_capacity("run", "s", 0, 0, 0, 1, now=1.0) is not None
        assert coord.reserve_capacity("run", "s", 0, 1, 0, 1, now=1.0) is not None
        assert coord.reserve_capacity("run", "s", 0, 2, 0, 1, now=1.0) is None  # 3rd file
        assert coord.reserved_files() == 2


def _stage_inline(coord, sid, sub_index, survivors, conn, now=100.0):
    """Stage ONE sub-stripe (no stripe completion) — reservation held, not published."""
    pb, size, sha = _canon(sid, sub_index, 0, 10, survivors)
    coord.ledger.record_substripe_result(
        "run", sid, 0, sub_index, conn.worker_id, 0, 10, len(survivors),
        remote_spool_path=None, size_bytes=size, sha256=sha, now=now)
    return coord.stage_inline_shard("run", sid, 0, sub_index, 0, 10, survivors,
                                    size, sha, now=now), size


def _stage_n_and_complete(coord, sid, conn, subs, now=100.0):
    """Stage N sub-stripes tiling a 30-seed stripe, complete + finalize -> publish.
    subs: list of (sub_index, seed_start, seed_count, survivors). Returns res list."""
    coord.ledger.set_trial_context("run", _ctx())
    out = []
    for sub_index, ss, sc, sv in subs:
        pb, size, sha = _canon(sid, sub_index, ss, sc, sv)
        coord.ledger.record_substripe_result(
            "run", sid, 0, sub_index, conn.worker_id, ss, sc, len(sv),
            remote_spool_path=None, size_bytes=size, sha256=sha, now=now)
        out.append(coord.stage_inline_shard("run", sid, 0, sub_index, ss, sc, sv,
                                            size, sha, now=now))
    total_sv = sum(len(sv) for _, _, _, sv in subs)
    coord.ledger.record_stripe_complete("run", sid, 0, conn.worker_id, len(subs), total_sv)
    coord.finalize_stripe("run", sid, now=now)
    return out


# ---------------------------------------------------------------------------
# GATE 25 — enqueue does NOT release capacity (L2)
# ---------------------------------------------------------------------------
def gate25_enqueue_does_not_release():
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord_staging(tmp, miner_stripe_size=1000)
        conn, sid = _assign_one(coord)
        res, size = _stage_complete_inline(coord, sid, conn, [[1, 0.9, None, [1]]])
        rid = res["reservation_id"]
        # published (enqueued) to Phase 5 but NOT yet acked -> reservation STILL held
        assert coord.ledger.get_shard("run", sid, 0, 0)["phase5_status"] == "enqueued"
        assert coord.reserved_bytes() == size and coord.reserved_files() == 1
        # attempting release without an ack does nothing (capacity stays counted)
        assert coord.release_after_ack("run", sid, 0, 0, rid, now=101.0) is False
        assert coord.reserved_bytes() == size


# ---------------------------------------------------------------------------
# GATE 26 — Phase 5 ack + local delete releases capacity (L2)
# ---------------------------------------------------------------------------
def gate26_ack_releases_capacity():
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord_staging(tmp, miner_stripe_size=1000)
        conn, sid = _assign_one(coord)
        res, size = _stage_complete_inline(coord, sid, conn, [[1, 0.9, None, [1]]])
        rid = res["reservation_id"]
        staged = res["staged_path"]
        # ack alone does not release
        coord.ack_shard("run", sid, 0, 0, now=101.0)
        assert coord.ledger.get_shard("run", sid, 0, 0)["phase5_status"] == "acked"
        assert coord.reserved_bytes() == size
        # ack + local delete releases; the local file is removed
        assert coord.release_after_ack("run", sid, 0, 0, rid, now=102.0) is True
        assert coord.reserved_bytes() == 0 and coord.reserved_files() == 0
        assert not os.path.isfile(staged)


# ---------------------------------------------------------------------------
# GATE 27 — high-water counts unacked staged files (L2)
# ---------------------------------------------------------------------------
def gate27_high_water_counts_unacked():
    with tempfile.TemporaryDirectory() as tmp:
        coord, (conn,) = _coord_workers(tmp, nvidia=15, worker_ids=("hostZ:gpu0",),
                                        staging_high_water_bytes=10 ** 9,
                                        staging_high_water_files=100)
        a = coord.assign_stripes("run", "java_lcg", 1, 30, [conn], now=100.0)[0]
        sid = a["stripe_id"]
        assert a["expected_substripes"] == 2
        res = _stage_n_and_complete(coord, sid, conn,
                                    [(0, 0, 15, [[1, 0.9, None, [1]]]),
                                     (1, 15, 15, [[2, 0.8, None, [2]]])])
        held = coord.reserved_bytes()
        assert coord.reserved_files() == 2 and held > 0   # both unacked shards count
        # tighten the mark to the current held amount: no headroom for a new transfer
        coord.config.staging_high_water_bytes = held
        assert coord.reserve_capacity("run", sid, 0, 9, 0, 1, now=200.0) is None
        # ack one (by event_id) -> its bytes free -> new transfer fits again
        assert coord.ack_by_event_id(res[0]["event_id"], now=201.0) is True
        assert coord.reserved_files() == 1
        assert coord.reserve_capacity("run", sid, 0, 9, 0, 1, now=202.0) is not None


# ---------------------------------------------------------------------------
# GATE 31 — durable remote-delete status; retried idempotently (SC1)
# ---------------------------------------------------------------------------
def gate31_durable_remote_delete():
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord_staging(tmp, miner_stripe_size=1000)
        conn, sid = _assign_one(coord)
        survivors = [[1, 0.9, 0, [1]]]
        pb, size, sha = _canon(sid, 0, 0, 10, survivors)
        remote = f"/var/spool/miner/{sid}/0.json"
        stub = _StubTransfer(payloads={remote: pb}, fail_delete=True)
        coord.transfer = stub
        coord.ledger.record_substripe_result(
            "run", sid, 0, 0, conn.worker_id, 0, 10, 1,
            remote_spool_path=remote, size_bytes=size, sha256=sha, now=100.0)
        res = coord.stage_remote_shard(conn, "run", sid, 0, 0, remote, size, sha, now=100.0)
        # verify SUCCEEDED even though delete failed; shard stays valid
        assert res["status"] == "verified"
        sh = coord.ledger.get_shard("run", sid, 0, 0)
        assert sh["staging_status"] == SH_VERIFIED
        assert sh["remote_delete_status"] == "failed" and sh["remote_delete_attempts"] == 1
        assert sh["remote_delete_error"]

        # idempotent retry, still failing -> attempts increments, shard still valid
        assert coord.retry_remote_delete(conn.node_config, "run", sid, 0, 0, remote, now=101.0) == "failed"
        sh = coord.ledger.get_shard("run", sid, 0, 0)
        assert sh["remote_delete_attempts"] == 2 and sh["staging_status"] == SH_VERIFIED

        # deletion later succeeds (a successful attempt still counts -> attempts=3)
        stub.fail_delete = False
        assert coord.retry_remote_delete(conn.node_config, "run", sid, 0, 0, remote, now=102.0) == "deleted"
        sh = coord.ledger.get_shard("run", sid, 0, 0)
        assert sh["remote_delete_status"] == "deleted" and sh["remote_deleted_at"] == 102.0
        assert sh["remote_delete_attempts"] == 3
        assert len(coord.ledger.get_shards("run", sid, 0)) == 1   # no duplicate shard

        # a further retry is a harmless no-op (already-deleted short-circuits; NO increment)
        assert coord.retry_remote_delete(conn.node_config, "run", sid, 0, 0, remote, now=103.0) == "deleted"
        assert coord.ledger.get_shard("run", sid, 0, 0)["remote_delete_attempts"] == 3


# ---------------------------------------------------------------------------
# GATE 32 — stale async-task fencing (L5)
# ---------------------------------------------------------------------------
def gate32_stale_async_task_fencing():
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord_staging(tmp, compute_lease_timeout=50.0)
        connA = _register(coord, "hostA:gpu0")
        connB = _register(coord, "hostB:gpu0")
        sid = "run_sX"
        survivors = [[1, 0.9, 0, [1]]]
        pb, size, sha = _canon(sid, 0, 0, 10, survivors)
        remote = f"/var/spool/miner/{sid}/0.json"
        coord.transfer = _StubTransfer(payloads={remote: pb})

        coord.ledger.add_stripe("run", sid, 0, 30, "java_lcg", 1, now=100.0)
        coord.ledger.claim_stripe("run", sid, "hostA:gpu0", 0, 3, lease_expires_at=150.0)
        connA.record_assignment(sid, 0)
        coord.ledger.record_substripe_result(
            "run", sid, 0, 0, "hostA:gpu0", 0, 10, 1,
            remote_spool_path=remote, size_bytes=size, sha256=sha, now=110.0)

        # attempt-0 fetch begins (in flight): reservation held, temp fetched
        task = coord.begin_remote_stage(connA, "run", sid, 0, 0, remote, size, sha, now=110.0)
        assert coord.reserved_bytes() == size
        gen0 = task.staging_generation

        # attempt-0 lease expires -> reclaim fences (gen++), attempt 1 -> worker B
        coord.ledger.reclaim_expired_leases("run", now=1000.0)
        coord.ledger.claim_stripe("run", sid, "hostB:gpu0", 1, 3, lease_expires_at=1100.0)
        connB.record_assignment(sid, 1)
        assert coord.ledger.get_stripe("run", sid)["staging_generation"] != gen0

        # attempt-0's fetch NOW completes -> callback finds its attempt inactive
        res = coord.finish_remote_stage(task, now=1001.0)
        assert res["status"] == "stale"
        # published nothing; released only its own reservation; removed its own files
        assert coord.enqueued == []
        assert coord.transfer.delete_calls == []
        assert coord.ledger.get_shard("run", sid, 0, 0)["staging_status"] != SH_VERIFIED
        assert coord.reserved_bytes() == 0 and coord.reserved_files() == 0
        assert not os.path.isfile(task.staged_path) and not os.path.isfile(task.temp_path)
        # attempt-1 ledger unchanged
        st = coord.ledger.get_stripe("run", sid)
        assert st["claimed_by"] == "hostB:gpu0" and st["current_attempt"] == 1


# ---------------------------------------------------------------------------
# GATE 36 — failure-path reservation cleanup on EVERY path (L8)
# ---------------------------------------------------------------------------
def gate36_failure_path_cleanup():
    survivors = [[1, 0.9, 0, [1]]]

    def _fresh(tmp, transfer=None, **cfg):
        coord = _coord_staging(tmp, transfer=transfer, **cfg)
        conn, sid = _assign_one(coord)
        return coord, conn, sid

    # (1) fetch exception
    with tempfile.TemporaryDirectory() as tmp:
        coord, conn, sid = _fresh(tmp, transfer=_StubTransfer(fail_fetch=True),
                                  miner_stripe_size=1000)
        pb, size, sha = _canon(sid, 0, 0, 10, survivors)
        remote = f"/var/spool/miner/{sid}/0.json"
        try:
            coord.stage_remote_shard(conn, "run", sid, 0, 0, remote, size, sha, now=100.0)
        except IOError:
            pass
        else:
            raise AssertionError("fetch failure must propagate")
        assert coord.reserved_bytes() == 0 and coord.reserved_files() == 0
        assert os.listdir(coord.config.staging_dir) == []

    # (2) hash mismatch
    with tempfile.TemporaryDirectory() as tmp:
        pb, size, sha = _canon("run_s0", 0, 0, 10, survivors)
        coord, conn, sid = _fresh(tmp, miner_stripe_size=1000)
        remote = f"/var/spool/miner/{sid}/0.json"
        coord.transfer = _StubTransfer(payloads={remote: pb + b"X"})
        try:
            coord.stage_remote_shard(conn, "run", sid, 0, 0, remote, size, sha, now=100.0)
        except StagingHashMismatch:
            pass
        assert coord.reserved_bytes() == 0 and os.listdir(coord.config.staging_dir) == []

    # (3) atomic-write failure (injected failing rename)
    with tempfile.TemporaryDirectory() as tmp:
        coord, conn, sid = _fresh(tmp, miner_stripe_size=1000)
        pb, size, sha = _canon(sid, 0, 0, 10, survivors)

        def _boom(src, dst):
            raise OSError("simulated atomic rename failure")
        coord._atomic_replace = _boom
        coord.ledger.record_substripe_result(
            "run", sid, 0, 0, conn.worker_id, 0, 10, 1,
            remote_spool_path=None, size_bytes=size, sha256=sha, now=100.0)
        try:
            coord.stage_inline_shard("run", sid, 0, 0, 0, 10, survivors, size, sha, now=100.0)
        except OSError:
            pass
        else:
            raise AssertionError("atomic-write failure must propagate")
        assert coord.reserved_bytes() == 0 and coord.reserved_files() == 0
        assert os.listdir(coord.config.staging_dir) == []

    # (4) staging timeout (cleanup primitive removes file then releases)
    with tempfile.TemporaryDirectory() as tmp:
        coord, conn, sid = _fresh(tmp, miner_stripe_size=1000)
        rid = coord.reserve_capacity("run", sid, 0, 0, 0, 50, now=100.0)
        tmpfile = os.path.join(coord.config.staging_dir, "sub0.tmp")
        os.makedirs(coord.config.staging_dir, exist_ok=True)
        open(tmpfile, "wb").write(b"partial")
        coord.ledger.set_reservation_paths(rid, temp_path=tmpfile)
        assert coord.reserved_bytes() == 50
        coord.cleanup_reservation(rid, mark_shard_failed=False, now=101.0)   # timeout cleanup
        assert coord.reserved_bytes() == 0 and not os.path.isfile(tmpfile)

    # (5) stale callback (L5) — reservation released, files gone
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord_staging(tmp, compute_lease_timeout=50.0)
        connA = _register(coord, "hostA:gpu0")
        sid = "run_sY"
        pb, size, sha = _canon(sid, 0, 0, 10, survivors)
        remote = f"/var/spool/miner/{sid}/0.json"
        coord.transfer = _StubTransfer(payloads={remote: pb})
        coord.ledger.add_stripe("run", sid, 0, 30, "java_lcg", 1, now=100.0)
        coord.ledger.claim_stripe("run", sid, "hostA:gpu0", 0, 3, lease_expires_at=150.0)
        connA.record_assignment(sid, 0)
        task = coord.begin_remote_stage(connA, "run", sid, 0, 0, remote, size, sha, now=110.0)
        coord.ledger.reclaim_expired_leases("run", now=1000.0)   # fence
        res = coord.finish_remote_stage(task, now=1001.0)
        assert res["status"] == "stale"
        assert coord.reserved_bytes() == 0 and coord.reserved_files() == 0
        assert not os.path.isfile(task.staged_path) and not os.path.isfile(task.temp_path)

    # (6) trial abort (cleanup primitive on a staged, held reservation)
    with tempfile.TemporaryDirectory() as tmp:
        coord, conn, sid = _fresh(tmp, miner_stripe_size=1000)
        res, size = _stage_inline(coord, sid, 0, survivors, conn)
        staged = res["staged_path"]
        assert os.path.isfile(staged) and coord.reserved_bytes() == size
        coord.cleanup_reservation(res["reservation_id"], now=200.0)   # abort cleanup
        assert coord.reserved_bytes() == 0 and coord.reserved_files() == 0
        assert not os.path.isfile(staged)


# ---------------------------------------------------------------------------
# Stage 4 — retry matrix, trial lifecycle, Phase5Sink (gates 7-12, 28, 33, 34)
# ---------------------------------------------------------------------------
def _coord_workers(tmp, sink=None, nvidia=5_000_000, nvidia_hybrid=2_500_000,
                   worker_ids=("hostA:gpu0", "hostB:gpu0"), **cfg):
    """Coordinator + registered workers whose ADVERTISED caps match the central
    config (else registration quarantines them). Set nvidia/nvidia_hybrid to
    control the effective sub-stripe cap (and thus expected_substripes)."""
    cfg.setdefault("miner_stripe_size", 1000)
    coord = _coord_staging(tmp, sink=sink, seed_cap_nvidia=nvidia,
                           seed_cap_nvidia_hybrid=nvidia_hybrid, **cfg)
    caps = {"amd": 2_000_000, "nvidia": nvidia,
            "amd_hybrid": 1_000_000, "nvidia_hybrid": nvidia_hybrid}
    conns = [_register(coord, wid, caps=caps) for wid in worker_ids]
    return coord, conns


# ---------------------------------------------------------------------------
# GATE 7 — partial-attempt cleanup before retry (B2)
# ---------------------------------------------------------------------------
def gate7_partial_attempt_cleanup():
    with tempfile.TemporaryDirectory() as tmp:
        sink = _StubSink()
        coord, (connA, connB) = _coord_workers(tmp, sink=sink, nvidia_hybrid=10)
        coord.ledger.create_trial("run", 1, now=100.0)
        a = coord.assign_stripes("run", "java_lcg_hybrid", 3, 30, [connA], now=100.0)[0]
        sid = a["stripe_id"]
        assert a["expected_substripes"] == 3   # hybrid cap 10 over 30 seeds
        # attempt-0 emits 2 GOOD shards (staged + verified)
        staged = []
        for i, (ss, sc) in enumerate([(0, 10), (10, 10)]):
            sv = [[i, 0.9, 0, [1]]]
            pb, size, sha = _canon(sid, i, ss, sc, sv)
            coord.ledger.record_substripe_result(
                "run", sid, 0, i, "hostA:gpu0", ss, sc, 1,
                remote_spool_path=None, size_bytes=size, sha256=sha, now=100.0)
            staged.append(coord.stage_inline_shard(
                "run", sid, 0, i, ss, sc, sv, size, sha, now=100.0)["staged_path"])
        assert coord.reserved_files() == 2 and all(os.path.isfile(p) for p in staged)

        # then a retryable failure (hybrid, first failure) -> reassign whole stripe
        act = coord.handle_stripe_failure(
            "run", sid, retryable=True, eligible_workers=[connA, connB], now=110.0)
        assert act["action"] == "reassigned" and act["worker_id"] == "hostB:gpu0"

        # attempt-0's local shards invalidated + removed; NOTHING published
        assert coord.enqueued == [] and sink.published == []
        assert coord.reserved_bytes() == 0 and coord.reserved_files() == 0
        assert all(not os.path.isfile(p) for p in staged)
        for i in range(2):
            assert coord.ledger.get_shard("run", sid, 0, i)["staging_status"] == SH_FAILED
        # stripe retried WHOLE as attempt 1 on a DIFFERENT worker, phase_degraded
        st = coord.ledger.get_stripe("run", sid)
        assert st["current_attempt"] == 1 and st["claimed_by"] == "hostB:gpu0"
        assert st["phase_degraded"] == 1 and st["state"] == ST_CLAIMED
        assert coord.ledger.get_trial("run")["state"] == "running"   # trial not failed


# ---------------------------------------------------------------------------
# GATE 8 — TrialCommit vs TrialAbort (B2)
# ---------------------------------------------------------------------------
def gate8_commit_vs_abort():
    # success -> TrialCommit publishes committed input
    with tempfile.TemporaryDirectory() as tmp:
        sink = _StubSink()
        coord = _coord_staging(tmp, sink=sink, miner_stripe_size=1000)
        coord.ledger.create_trial("run", 1, now=100.0)
        conn, sid = _assign_one(coord)
        _stage_complete_inline(coord, sid, conn, [[1, 0.9, None, [1]]])
        assert coord.ledger.get_stripe("run", sid)["state"] == ST_DONE
        assert len(sink.published) == 1
        coord.commit_trial("run", now=200.0)
        assert coord.ledger.get_trial("run")["state"] == "committed"
        assert len(sink.commits) == 1 and sink.aborts == []

    # terminal failure -> TrialAbort, provisional discarded, commit refused
    with tempfile.TemporaryDirectory() as tmp:
        sink = _StubSink()
        coord = _coord_staging(tmp, sink=sink, miner_stripe_size=1000)
        coord.ledger.create_trial("run", 1, now=100.0)
        conn, sid = _assign_one(coord)
        _stage_complete_inline(coord, sid, conn, [[1, 0.9, None, [1]]])
        coord.abort_trial("run", reason="terminal", now=200.0)
        assert coord.ledger.get_trial("run")["state"] == "aborted"
        assert len(sink.aborts) == 1
        try:
            coord.commit_trial("run", now=201.0)
        except TrialAborted:
            pass
        else:
            raise AssertionError("TrialCommit after abort must be refused")
        assert sink.commits == []


# ---------------------------------------------------------------------------
# GATE 9 — Phase 1/2 (constant) immediate failure (B3)
# ---------------------------------------------------------------------------
def gate9_constant_phase_immediate_fail():
    with tempfile.TemporaryDirectory() as tmp:
        sink = _StubSink()
        coord, (connA, connB) = _coord_workers(tmp, sink=sink)
        coord.ledger.create_trial("run", 1, now=100.0)
        a = coord.assign_stripes("run", "java_lcg", 1, 30, [connA], now=100.0)[0]  # phase 1
        sid = a["stripe_id"]
        # even a "retryable" failure fails closed in a constant phase
        act = coord.handle_stripe_failure(
            "run", sid, retryable=True, eligible_workers=[connA, connB], now=110.0)
        assert act["action"] == "fail_trial" and act["reason"] == "constant_phase"
        assert coord.ledger.get_trial("run")["state"] == "aborted" and len(sink.aborts) == 1
        # NO retry: attempt not incremented, stripe cancelled
        st = coord.ledger.get_stripe("run", sid)
        assert st["current_attempt"] == 0 and st["state"] == ST_CANCELLED


# ---------------------------------------------------------------------------
# GATE 10 — Phase 3/4 (hybrid) one-retry-then-fail (B3)
# ---------------------------------------------------------------------------
def gate10_hybrid_one_retry_then_fail():
    with tempfile.TemporaryDirectory() as tmp:
        sink = _StubSink()
        coord, (connA, connB) = _coord_workers(tmp, sink=sink)
        coord.ledger.create_trial("run", 1, now=100.0)
        a = coord.assign_stripes("run", "java_lcg_hybrid", 3, 30, [connA], now=100.0)[0]
        sid = a["stripe_id"]
        # first retryable failure -> reassign ONCE to a DIFFERENT eligible worker
        act1 = coord.handle_stripe_failure(
            "run", sid, retryable=True, eligible_workers=[connA, connB], now=110.0)
        assert act1["action"] == "reassigned" and act1["worker_id"] == "hostB:gpu0"
        st = coord.ledger.get_stripe("run", sid)
        assert st["current_attempt"] == 1 and st["claimed_by"] == "hostB:gpu0"
        assert st["phase_degraded"] == 1
        assert coord.ledger.get_trial("run")["state"] == "running"
        # second retryable failure -> fail trial
        act2 = coord.handle_stripe_failure(
            "run", sid, retryable=True, eligible_workers=[connA, connB], now=120.0)
        assert act2["action"] == "fail_trial" and act2["reason"] == "hybrid_second_failure"
        assert coord.ledger.get_trial("run")["state"] == "aborted"


# ---------------------------------------------------------------------------
# GATE 11 — retryable=False immediate failure; retry NOT consumed (B3)
# ---------------------------------------------------------------------------
def gate11_non_retryable_immediate_fail():
    with tempfile.TemporaryDirectory() as tmp:
        sink = _StubSink()
        coord, (connA, connB) = _coord_workers(tmp, sink=sink)
        coord.ledger.create_trial("run", 1, now=100.0)
        # hybrid phase, but retryable=False -> fail immediately, retry not consumed
        a = coord.assign_stripes("run", "java_lcg_hybrid", 3, 30, [connA], now=100.0)[0]
        sid = a["stripe_id"]
        act = coord.handle_stripe_failure(
            "run", sid, retryable=False, eligible_workers=[connA, connB], now=110.0)
        assert act["action"] == "fail_trial" and act["reason"] == "non_retryable"
        assert coord.ledger.get_trial("run")["state"] == "aborted"
        assert coord.ledger.get_stripe("run", sid)["current_attempt"] == 0   # not consumed


# ---------------------------------------------------------------------------
# GATE 12 — lease expiry applies the phase-specific policy (B3)
# ---------------------------------------------------------------------------
def gate12_lease_expiry_policy():
    # constant phase expiry -> fail trial
    with tempfile.TemporaryDirectory() as tmp:
        sink = _StubSink()
        coord, (connA, connB) = _coord_workers(tmp, sink=sink, compute_lease_timeout=50.0)
        coord.ledger.create_trial("run", 1, now=100.0)
        a = coord.assign_stripes("run", "java_lcg", 1, 30, [connA], now=100.0)[0]
        out = coord.process_lease_expiry("run", [connA, connB], now=1000.0)
        assert out and out[0]["action"] == "fail_trial"
        assert coord.ledger.get_trial("run")["state"] == "aborted"

    # hybrid phase expiry -> reassign once
    with tempfile.TemporaryDirectory() as tmp:
        sink = _StubSink()
        coord, (connA, connB) = _coord_workers(tmp, sink=sink, compute_lease_timeout=50.0)
        coord.ledger.create_trial("run", 1, now=100.0)
        a = coord.assign_stripes("run", "java_lcg_hybrid", 3, 30, [connA], now=100.0)[0]
        sid = a["stripe_id"]
        out = coord.process_lease_expiry("run", [connA, connB], now=1000.0)
        assert out and out[0]["action"] == "reassigned" and out[0]["worker_id"] == "hostB:gpu0"
        st = coord.ledger.get_stripe("run", sid)
        assert st["current_attempt"] == 1 and st["phase_degraded"] == 1
        assert coord.ledger.get_trial("run")["state"] == "running"


# ---------------------------------------------------------------------------
# GATE 28 — whole-trial abort (L3) + L7 retain-on-failure
# ---------------------------------------------------------------------------
def gate28_whole_trial_abort():
    with tempfile.TemporaryDirectory() as tmp:
        sink = _StubSink()
        coord = _coord_staging(tmp, sink=sink, miner_stripe_size=1000)
        coord.ledger.set_trial_context("run", _ctx())
        coord.ledger.create_trial("run", 1, now=100.0)
        conn = _register(coord, "hostA:gpu0")
        staged = []
        for sid in ("run_s1", "run_s2"):
            coord.ledger.add_stripe("run", sid, 0, 30, "java_lcg", 1, now=100.0)
            coord.ledger.claim_stripe("run", sid, "hostA:gpu0", 0, 1, lease_expires_at=1e9)
            conn.record_assignment(sid, 0)
            sv = [[1, 0.9, None, [1]]]
            pb, size, sha = _canon(sid, 0, 0, 30, sv)
            coord.ledger.record_substripe_result(
                "run", sid, 0, 0, "hostA:gpu0", 0, 30, 1,
                remote_spool_path=None, size_bytes=size, sha256=sha, now=100.0)
            staged.append(coord.stage_inline_shard(
                "run", sid, 0, 0, 0, 30, sv, size, sha, now=100.0)["staged_path"])
            coord.ledger.record_stripe_complete("run", sid, 0, "hostA:gpu0", 1, 1)
            coord.finalize_stripe("run", sid, now=100.0)
        assert len(sink.published) == 2 and coord.reserved_files() == 2

        # a third stripe fails terminally -> abort routed OFF the dispatch loop
        coord.ledger.add_stripe("run", "run_s3", 0, 30, "java_lcg", 1, now=100.0)
        coord.ledger.claim_stripe("run", "run_s3", "hostA:gpu0", 0, 1, lease_expires_at=1e9)
        r = coord.submit_abort("run", reason="s3 terminal", now=200.0).result()
        assert r["cleanup"] == "done" and r["first"] is True
        assert coord.ledger.get_trial("run")["state"] == "aborted" and len(sink.aborts) == 1
        # all provisional shards invalidated; no leaked files; reservations released
        assert coord.reserved_bytes() == 0 and coord.reserved_files() == 0
        assert all(not os.path.isfile(p) for p in staged)
        assert coord.ledger.get_stripe("run", "run_s3")["state"] == ST_CANCELLED
        # no committed input; subsequent TrialCommit refused
        try:
            coord.commit_trial("run", now=201.0)
        except TrialAborted:
            pass
        else:
            raise AssertionError("TrialCommit after abort must be refused")
        assert sink.commits == []

    # L7 retain-on-failure: if the sync sink.abort_trial raises, files + reservations
    # are RETAINED (never deleted merely because delivery was attempted); retried.
    with tempfile.TemporaryDirectory() as tmp:
        sink = _StubSink(abort_raises=True)
        coord = _coord_staging(tmp, sink=sink, miner_stripe_size=1000)
        coord.ledger.create_trial("run", 1, now=100.0)
        conn = _register(coord, "hostA:gpu0")
        sid = "run_s1"
        coord.ledger.add_stripe("run", sid, 0, 30, "java_lcg", 1, now=100.0)
        coord.ledger.claim_stripe("run", sid, "hostA:gpu0", 0, 1, lease_expires_at=1e9)
        sv = [[1, 0.9, None, [1]]]
        pb, size, sha = _canon(sid, 0, 0, 30, sv)
        coord.ledger.record_substripe_result(
            "run", sid, 0, 0, "hostA:gpu0", 0, 30, 1,
            remote_spool_path=None, size_bytes=size, sha256=sha, now=100.0)
        staged = coord.stage_inline_shard("run", sid, 0, 0, 0, 30, sv, size, sha, now=100.0)["staged_path"]
        res = coord.abort_trial("run", reason="x", now=200.0)
        assert res["cleanup"] == "failed"
        assert os.path.isfile(staged) and coord.reserved_files() == 1     # RETAINED
        assert coord.ledger.get_trial("run")["state"] == "aborted"
        assert coord.ledger.get_trial("run")["abort_cleanup_status"] == "failed"
        # idempotent retry with a now-working sink completes cleanup
        sink.abort_raises = False
        res2 = coord.abort_trial("run", reason="x", now=201.0)
        assert res2["cleanup"] == "done" and res2["first"] is False
        assert not os.path.isfile(staged) and coord.reserved_files() == 0
        assert len(sink.aborts) == 2


# ---------------------------------------------------------------------------
# GATE 33 — event-id ack idempotency (L6)
# ---------------------------------------------------------------------------
def gate33_event_id_ack_idempotency():
    with tempfile.TemporaryDirectory() as tmp:
        coord, (conn,) = _coord_workers(tmp, nvidia=15, worker_ids=("hostZ:gpu0",))
        a = coord.assign_stripes("run", "java_lcg", 1, 30, [conn], now=100.0)[0]
        sid = a["stripe_id"]
        res = _stage_n_and_complete(coord, sid, conn,
                                    [(0, 0, 15, [[1, 0.9, None, [1]]]),
                                     (1, 15, 15, [[2, 0.8, None, [2]]])])
        evA, evB = res[0]["event_id"], res[1]["event_id"]
        assert coord.reserved_files() == 2 and evA != evB
        # two acks for the SAME event_id release its reservation exactly once
        assert coord.ack_by_event_id(evA, now=101.0) is True
        assert coord.ack_by_event_id(evA, now=102.0) is False   # idempotent no-op
        assert coord.reserved_files() == 1                       # only A released
        # an ack for event A never touched event B
        assert coord.ledger.get_shard("run", sid, 0, 1)["phase5_status"] == "enqueued"
        assert coord.ack_by_event_id(evB, now=103.0) is True
        assert coord.reserved_files() == 0


# ---------------------------------------------------------------------------
# GATE 34 — abort-discard race: local file remains until sync abort returns (L7)
# ---------------------------------------------------------------------------
def gate34_abort_discard_race():
    with tempfile.TemporaryDirectory() as tmp:
        holder = {}

        def probe():
            # the shard's local file MUST still exist while abort_trial executes
            holder["existed_during_abort"] = os.path.isfile(holder["staged"])

        sink = _StubSink(abort_probe=probe)
        coord = _coord_staging(tmp, sink=sink, miner_stripe_size=1000)
        coord.ledger.create_trial("run", 1, now=100.0)
        conn = _register(coord, "hostA:gpu0")
        sid = "run_s1"
        coord.ledger.add_stripe("run", sid, 0, 30, "java_lcg", 1, now=100.0)
        coord.ledger.claim_stripe("run", sid, "hostA:gpu0", 0, 1, lease_expires_at=1e9)
        sv = [[1, 0.9, None, [1]]]
        pb, size, sha = _canon(sid, 0, 0, 30, sv)
        coord.ledger.record_substripe_result(
            "run", sid, 0, 0, "hostA:gpu0", 0, 30, 1,
            remote_spool_path=None, size_bytes=size, sha256=sha, now=100.0)
        # unacked, actively-consumed shard: staged + held, NOT acked
        holder["staged"] = coord.stage_inline_shard(
            "run", sid, 0, 0, 0, 30, sv, size, sha, now=100.0)["staged_path"]
        assert os.path.isfile(holder["staged"]) and coord.reserved_files() == 1

        coord.abort_trial("run", reason="race", now=200.0)
        # the local file REMAINED while the synchronous stub executed...
        assert holder["existed_during_abort"] is True
        # ...and ONLY AFTER the stub returned was it deleted + its reservation released
        assert not os.path.isfile(holder["staged"])
        assert coord.reserved_files() == 0


# ---------------------------------------------------------------------------
# Stage 5 — integration, coexistence, non-regression (gates 20-23)
# ---------------------------------------------------------------------------
def gate20_resolver_end_to_end():
    """End-to-end through a REAL worker daemon using the REAL Stage-0-patched
    ResidueResolver: a wrong dataset_sha256 -> stripe_error(retryable=False) ->
    the coordinator applies matrix row 4 (fail trial immediately)."""
    with tempfile.TemporaryDirectory() as tmp:
        ds = os.path.join(tmp, "dataset.json")
        with open(ds, "w") as f:
            f.write('[{"draw":1},{"draw":2},{"draw":3}]')

        sink = _StubSink()
        coord = _coord_staging(tmp, sink=sink, miner_stripe_size=1000)
        coord.ledger.create_trial("run", 1, now=100.0)

        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        srv.settimeout(10)
        port = srv.getsockname()[1]

        worker = RangeMinerWorker(
            host="127.0.0.1", port=port, gpu_id=0, caps=VramCaps(),
            executor=SieveExecutor(resolver=ResidueResolver(), device_index=0).execute,
            gpu_info=GpuInfo("cuda", "stub", 12 * 1024 ** 3),
            heartbeat_interval=999, miner_output_dir=tmp)
        errbox = {}

        def run():
            try:
                worker.connect()
                worker.register()
                worker.serve_forever()
            except Exception:
                errbox["e"] = traceback.format_exc()

        t = threading.Thread(target=run, daemon=True)
        t.start()
        conn_sock, _ = srv.accept()
        fs = MinerFramedSocket(conn_sock)
        try:
            reg = fs.recv_msg()
            assert reg.message_type == "register"
            wid = reg.worker_id
            # coordinator binds the connection + validates advertised caps (eligible)
            node = NodeConfig(hostname=reg.hostname, spool_root="/var/spool/miner",
                              ssh_address="10.0.0.9", ssh_user="michael")
            wconn = coord.register_worker(
                worker_id=wid, hostname=reg.hostname, backend=reg.backend,
                capabilities=reg.capabilities, node_config=node, now=100.0)
            assert not wconn.quarantined

            sid = "run_s0"
            coord.ledger.add_stripe("run", sid, 0, 30, "java_lcg", 1, now=100.0)
            coord.ledger.claim_stripe("run", sid, wid, 0, 1, lease_expires_at=1e9)
            wconn.record_assignment(sid, 0)

            # payload carries BOTH mandatory hashes; dataset_sha256 is deliberately WRONG
            payload = coord.build_stripe_assign_payload(
                dataset_path=ds, window_size=3, sessions=None, offset=0,
                residues=[1, 2, 3], dataset_sha256="deadbeef_wrong",
                # [S172 D6] phase + both directional thresholds are REQUIRED
                # keyword-only args — a payload can no longer be built without
                # an explicit, direction-resolved threshold.
                phase=1, forward_threshold=0.31, reverse_threshold=0.47)
            assert payload["dataset_sha256"] == "deadbeef_wrong"
            assert payload["residue_sha256"]      # residue_sha256 is mandatory, present
            fs.send_msg(StripeAssignMessage(
                stripe_id=sid, prng_type="java_lcg", family_name="java_lcg",
                seed_start=0, seed_count=30, phase=1, payload=payload))

            resp = fs.recv_msg()
            # the REAL patched resolver rejected the dataset hash, non-retryably
            assert resp.message_type == "stripe_error", resp.message_type
            assert resp.retryable is False
            assert "dataset_sha256" in resp.error

            # coordinator: identity/fence gate, then matrix row 4 -> fail trial
            ok, reason = coord.accept_stripe_message(
                wconn, "run", sid, resp.worker_id, (ST_CLAIMED, ST_STAGING))
            assert ok, reason
            act = coord.handle_stripe_failure(
                "run", sid, retryable=resp.retryable, eligible_workers=[wconn], now=110.0)
            assert act["action"] == "fail_trial" and act["reason"] == "non_retryable"
            assert coord.ledger.get_trial("run")["state"] == "aborted"
            assert len(sink.aborts) == 1

            fs.send_msg(MinerShutdownMessage())
            t.join(timeout=5)
        finally:
            fs.close()
            srv.close()
        assert "e" not in errbox, errbox.get("e")


def gate21_no_phase5_assembly():
    """The coordinator module owns NO Phase-5 assembly: no 22-array build, no
    dedup/ordering, no NPZ contract wall, no serial np.load collection (§3.A)."""
    src = open(COORD.__file__, "r", encoding="utf-8").read()
    # Concrete assembly-CODE tokens only (prose like "must NOT run the contract
    # wall / dedup" legitimately appears in the ownership comments — §3.A).
    forbidden = [
        "range_miner_npz_writer",   # the Phase-5 writer module
        "EXPECTED_NPZ_KEYS",        # the NPZ contract wall
        "np.savez", "np.load",      # serial NPZ collection / write
        "import numpy",             # no array assembly at all
        "process_sharded",          # a Phase-5 backend
    ]
    for tok in forbidden:
        assert tok not in src, f"coordinator must not reference {tok!r} (Phase 5 owns it)"
    # importing the coordinator pulls in no Phase-5 assembly module + defines none
    assert "miner.range_miner_npz_writer" not in sys.modules
    assert not hasattr(COORD, "np") and not hasattr(COORD, "numpy")
    assert not hasattr(COORD, "assemble_arrays") and not hasattr(COORD, "run_contract_wall")


def gate22_coexistence():
    """use_range_miner selects the miner path (run_trial_miner builds + drives a
    real coordinator); PWC + ZMQ remain importable + unmodified; only the miner
    code deliverables changed."""
    # 1) only the miner/Phase-4 .py deliverables are changed (git); PWC/ZMQ untouched
    r = subprocess.run(["git", "status", "--porcelain"], cwd=_ROOT,
                       capture_output=True, text=True)
    changed_py = {ln[3:].strip() for ln in r.stdout.splitlines()
                  if ln[3:].strip().endswith(".py")}
    allowed = {
        "miner/range_miner_coordinator.py", "miner/range_miner_worker.py",
        "tests/test_s172_phase3_worker.py", "tests/test_s172_phase4_coordinator.py",
        # Phase-1 gate 2 asserted run_trial_miner raises NotImplementedError;
        # Beta's binding serve-path ruling ordered that raise DELETED, so the one
        # Phase-1 gate that tested it was necessarily updated (flagged for review).
        "tests/test_s172_phase1_scaffolding.py",
        # Correction-2 Defect 6: the real _use_miner integration call is wired
        # (unique run_id + resolved phase/window/staging/bind). This edit lives
        # ENTIRELY inside the `use_range_miner` gate — the PWC and ZMQ call paths
        # in this file are untouched, so coexistence holds.
        "window_optimizer_integration_final.py",
        # Phase-5 D0 (metadata-seam + durable-context correction): the D0 change
        # touches only the miner coordinator + the `use_range_miner` call site
        # (both already allowed above); it also ships its own acceptance harness.
        # Listing that harness here keeps this coexistence whitelist truthful —
        # PWC/ZMQ/pwc_protocol remain untouched (flagged for review).
        "tests/test_s172_phase5_d0.py",
        # Phase-5 D1.0 (workflow bidirectionality + abort/commit terminal-race
        # correction): the change touches only the miner coordinator (already
        # allowed above) and ships its own acceptance harness. Listed here for the
        # same reason as D0's — PWC/ZMQ/pwc_protocol remain untouched (flagged for
        # review).
        "tests/test_s172_phase5_d1_workflow.py",
        # Phase-5 D1.1 (four-population assembly engine + the concrete
        # AssemblingPhase5Sink): a NEW Phase-5 module plus its own acceptance
        # harness. It imports the miner coordinator but changes nothing in it —
        # PWC/ZMQ/pwc_protocol remain untouched (flagged for review).
        "miner/range_miner_npz_writer.py",
        "tests/test_s172_phase5_d1_engine.py",
        # Phase-5 D2 (directional uniqueness at BOTH enforcement layers): a NEW
        # acceptance harness ONLY — it adds no production change (D2-A drives the
        # unchanged serve path; D2-B probes the unchanged D1.1 writer). Registered
        # here per the extended Team Beta standing whitelist rule so the one new
        # untracked test path does not red this coexistence gate; PWC/ZMQ/
        # pwc_protocol remain untouched (flagged for review).
        "tests/test_s172_phase5_d2_directional_uniqueness.py",
        # Phase-5 D3.0 (legacy seam correction: canonical PRNG/skip encoding +
        # rectangular 22-array empty output). This deliverable DOES change
        # production files — two of them — so both are registered explicitly:
        #   * convert_survivors_to_binary.py — the local 12-entry
        #     PRNG_TYPE_ENCODING / SKIP_MODE_ENCODING tables are deleted in
        #     favour of utils/prng_encoding, and the empty case now writes all
        #     22 zero-length arrays instead of one.
        #   * window_optimizer_integration_final.py — the inline _PRNG_ENC /
        #     _SKIP_ENC tables in the S145-R1 accumulator seam are likewise
        #     replaced by the canonical encoders. Already allowed above for the
        #     Correction-2 Defect 6 wiring; the D3.0 edit is confined to that
        #     ~12-line encoding seam (the merge, supersede, backfill, sort and
        #     dual-write logic are untouched).
        # PWC/ZMQ/pwc_protocol remain unmodified, so coexistence still holds
        # (flagged for review).
        "convert_survivors_to_binary.py",
        "tests/test_s172_phase5_d3_0_encoding_contract.py",
        # Phase-5 D3 (shared backend-neutral 24 -> 22 columnizer + independent
        # structural validator): a NEW module under utils/ plus its own
        # acceptance harness. It is deliberately NOT wired into any live path —
        # the existing `_survivors_to_arrays` closure and the
        # convert_survivors_to_binary array block stay in place and in use until
        # D3.5 — so no producer or accumulator call site is rewired here.
        # PWC/ZMQ/pwc_protocol remain untouched (flagged for review).
        "utils/canonical_arrays.py",
        "tests/test_s172_phase5_d3_columnizer.py",
        # Phase-5 D3.25 (mode-preserving backend result contract + canonical
        # candidate-ingress normalization). This is the deliverable that ENDS
        # the "PWC/ZMQ are untouched" era, by design and by approved scope
        # (D3.25 REV3 §6 lists persistent_worker_coordinator.py and
        # zmq_sqlite_coordinator.py as may-modify):
        #   * utils/canonical_records.py — NEW: the one shared canonical record
        #     builder (`_mode_records` extracted out of the miner package) plus
        #     the `step1_trial_populations_v2` producer contract.
        #   * persistent_worker_coordinator.py / zmq_sqlite_coordinator.py —
        #     both now return all four directional maps under the v2 schema
        #     stamp, on EVERY return path including their pruned early returns,
        #     and both egress-validate before returning. Previously each
        #     returned only the CONSTANT map pair and structurally discarded
        #     the two variable maps, which is precisely the defect D3.25 fixes.
        #   * miner/range_miner_npz_writer.py (already allowed above) — now
        #     imports the extracted builder instead of defining it.
        #   * window_optimizer_integration_final.py (already allowed above) —
        #     the adapter's cross-mode set union is replaced by per-mode
        #     normalization behind an ingress validation wall.
        "utils/canonical_records.py",
        "persistent_worker_coordinator.py",
        "zmq_sqlite_coordinator.py",
        "tests/test_s172_phase5_d3_25_candidate_ingress.py",
        # Phase-5 D3.5 (the shared run finalizer: L2 winner selection, L3
        # array-domain merge, immutable-generation publication):
        #   * utils/run_finalizer.py — NEW: the one finalizer every backend
        #     goes through. It reuses D3's `records_to_arrays` /
        #     `validate_array_bundle` and reimplements neither.
        #   * window_optimizer_integration_final.py (already allowed above) —
        #     the inline S145-R1 NPZ-accumulator block is replaced by a call to
        #     that finalizer, the legacy score-only `deduplicate_survivors` is
        #     removed along with the `convert_survivors_to_binary.py` subprocess
        #     fallback, and the call now sits OUTSIDE the broad swallow wrapper
        #     so a canonical-finalization failure propagates (D3.5 §11 [B4]).
        # PWC/ZMQ call paths and pwc_protocol remain untouched by this
        # deliverable (flagged for review).
        "utils/run_finalizer.py",
        "tests/test_s172_phase5_d3_5_finalizer.py",
        # D3.5 MIGRATION-GATE RETIREMENT CAUSED BY REMOVAL OF THE LIVE INLINE
        # WRITER (Team Beta ruling on harness expiry). Both harnesses below are
        # ALREADY whitelisted above for their own deliverables; they are named a
        # second time here because D3.5 edits them for a reason that belongs to
        # D3.5, not to D3 or D3.0, and that reason must be legible in this list:
        #
        #   * tests/test_s172_phase5_d3_columnizer.py — C8 drove TWO legacy
        #     columnizers. The second was the inline `_survivors_to_arrays`
        #     closure in window_optimizer_integration_final.py, extracted from
        #     live source by AST line-range. D3.5 replaced the inline
        #     run-finalization block, so that closure no longer exists and the
        #     extraction aborted the harness at import. The inline half of C8 is
        #     REMOVED; parity against the standalone convert_survivors_to_binary
        #     writer is unchanged. C1-C7 and C9-C10 (incl. mutation evidence)
        #     are untouched. Tally stays 10/10.
        #
        #   * tests/test_s172_phase5_d3_0_encoding_contract.py — the same
        #     extraction fed SEVEN consumers (E1-E7, E9, E10). The machinery is
        #     REMOVED; E1-E7 now assert against the standalone converter and,
        #     where the check is about the encoding contract itself rather than
        #     one writer, additionally against the canonical utils/prng_encoding
        #     entry points. E8 is unchanged. E9 compares only the standalone
        #     converter against the hand-written golden. E10 is reframed to the
        #     standalone converter's mixed java_lcg constant/hybrid column.
        #     Tally stays 10/10.
        #
        # This is a PLANNED EXPIRY, not drift: D3's own module docstring stated
        # the inline closure would "stay in place and in use UNTIL D3.5". The
        # closure is NOT retained as dead production code, its source is NOT
        # snapshotted into either harness, and no failing check was skipped to
        # preserve a tally. utils/canonical_arrays.py, utils/prng_encoding.py
        # and convert_survivors_to_binary.py are byte-identical to 70cd6f0.
        #
        # Phase-5 D4 (the two-backend assembly interface + `serial_reference`):
        #   * miner/assembly_backends.py — NEW: `ASSEMBLY_BACKENDS`,
        #     `get_assembly_backend` (fail-closed; `process_sharded` is DECLARED
        #     but raises NotImplementedError naming D5), the frozen
        #     `BackendAssemblyResult` / `AssemblyMeasurement` return contract,
        #     and `SerialReferenceBackend` — a THIN wrapper that delegates to
        #     D1.1's `assemble_trial` and measures the call. It writes no
        #     assembly, columnization, dedup, ordering or publication logic:
        #     every one of those already exists and D4 calls it (gate G7 proves
        #     that at AST level). It deliberately does NOT import
        #     utils/run_finalizer — a backend produces a MinerTrialAssembly and
        #     stops.
        #   * tests/test_s172_phase5_d4_serial_backend.py — its own acceptance
        #     harness (G1-G8, 9 mutants).
        # No existing production module is touched, so this is a pure ADD:
        # PWC/ZMQ/pwc_protocol remain untouched by it (flagged for review).
        "miner/assembly_backends.py",
        "tests/test_s172_phase5_d4_serial_backend.py",
        # Phase-5 D5 (the `process_sharded` assembly backend + the
        # semantics-preserving D1.1 extraction it plugs into). Registered here
        # per the extended Team Beta standing whitelist rule already applied to
        # D2/D3/D3.25/D3.5/D4 above, so D5's new paths do not red this
        # coexistence gate. Two commits:
        #   * COMMIT 1 — miner/range_miner_npz_writer.py (already allowed
        #     above): `assemble_trial` is refactored into a thin serial wrapper
        #     over three units extracted VERBATIM from its own body —
        #     `prepare_trial_assembly` (the §5.1/§5.2/§5.4 metadata gauntlet +
        #     the deterministic spool order), the now-public
        #     `read_and_validate_spool` (returning the new
        #     `ValidatedSpoolProjection` instead of the parsed payload), and
        #     `merge_validated_spools` (the §5.4 + §5.5/§6 global assembly).
        #     ZERO behavioural change; D1.1 staying 18/18 with no test edit is
        #     the proof.
        #   * COMMIT 2 — miner/assembly_shard_worker.py (NEW): the CPU-only
        #     per-spool worker, the `allow_pickle=False` projection artifact
        #     codec, the §5 sampled concurrent-tree RSS sampler, and the
        #     parent-side orchestration. miner/assembly_backends.py (already
        #     allowed above) gains the thin `ProcessShardedBackend`; its
        #     `get_assembly_backend` name-only resolution still raises
        #     NotImplementedError naming D5, so D4's G4 and its M2 mutant are
        #     untouched.
        # No PWC/ZMQ/pwc_protocol path is touched by either commit, so
        # coexistence still holds (flagged for review).
        "miner/assembly_shard_worker.py",
        "tests/test_s172_phase5_d5_process_sharded.py",
        # Phase-5 D6 (the production integration adapter + the Zeus single-GPU
        # certified-generation smoke). Registered here per the same extended
        # Team Beta standing whitelist rule applied to D2/D3/D3.25/D3.5/D4/D5:
        #   * miner/step1_ingress.py — NEW: the one seam between a committed
        #     miner trial and the legacy Step-1 shapes. It resolves the assembly
        #     backend (default `serial_reference`), builds the sink around it,
        #     fetches the STORED MinerTrialAssembly fail-closed, and appends
        #     `canonical_records_constant`/`_variable` into the accumulator
        #     WITHOUT re-normalization. It imports neither utils/canonical_records
        #     nor any PWC/ZMQ module — routing miner output through the D3.25
        #     ingress wall is what REV3 §4 forbids.
        #   * miner/range_miner_npz_writer.py (already allowed above) —
        #     `AssemblingPhase5Sink` gains ONE optional constructor argument,
        #     `backend=None`, and a `_assemble` seam. `None` keeps the pre-D6
        #     behaviour verbatim (`assemble_trial`), which is why D1.1 stays
        #     18/18 with no test edit.
        #   * window_optimizer_integration_final.py (already allowed above) —
        #     the edit is confined to the `use_range_miner` gate and to
        #     `_build_test_result_from_miner`: the gate now builds the Phase-5
        #     sink and passes it to run_trial_miner (it passed None before, so
        #     no assembly was ever produced), and the builder ingests the stored
        #     assembly instead of returning +0. `_build_test_result_from_pw`,
        #     the PWC and ZMQ gates, and `_flush_npz_incremental` are all
        #     byte-identical to 2a6e0f8 — D6's own G-TESTRESULT and
        #     G-FLUSH-CADENCE assert exactly that against `git show`.
        #   * tests/test_s172_phase5_d6_production_adapter.py — the 3.A
        #     acceptance harness (7 gates + 16 mutants).
        #   * tests/smoke_s172_phase5_d6_zeus_single_gpu.py — the 3.B
        #     real-silicon smoke. It drives the production `use_range_miner`
        #     call against a REAL `miner/range_miner_worker.py` process on the
        #     passed-through RTX 3080 Ti; it is run by hand, not by this gate.
        #   * tests/test_s172_phase5_d3_25_candidate_ingress.py (already
        #     allowed above) — G13 and its three bite proofs are updated from
        #     the interim "miner appends nothing / +0" contract to the D6 one.
        #     D3.25's own note read "miner both-mode run-level candidate output
        #     uncertified until D6"; D6 certifies it, and what G13 still guards
        #     — PWC/ZMQ isolation, flush cadence, never a fabricated zero — is
        #     unchanged and strengthened.
        # PWC/ZMQ/pwc_protocol remain unmodified, so coexistence still holds
        # (flagged for review).
        "miner/step1_ingress.py",
        "tests/test_s172_phase5_d6_production_adapter.py",
        "tests/smoke_s172_phase5_d6_zeus_single_gpu.py",
        # Phase-5 D6 CORRECTION PASS (Team Beta's threshold blocker). The
        # correction adds ONE more changed deliverable beyond the D6 set above:
        #   * miner/range_miner_protocol.py — two ADDITIVE, defaulted envelope
        #     fields (`SubStripeResultMessage.effective_threshold`,
        #     `StripeCompleteMessage.effective_threshold`) carrying the
        #     effective-threshold provenance leg. Additive-with-default keeps the
        #     Phase-2 framing contract intact: every dataclass field still has a
        #     default, and `from_dict` still filters unknown kwargs, so a pre-D6
        #     peer decodes unchanged. No message type, no field rename, no
        #     removal — and `persistent/pwc_protocol.py` is NOT touched.
        #   * miner/range_miner_coordinator.py, miner/range_miner_worker.py and
        #     window_optimizer_integration_final.py (all already allowed above) —
        #     the single threshold chokepoint, the worker-side contract
        #     validation + effective-threshold reporting, and the shared-authority
        #     residue derivation at the `use_range_miner` call site.
        #   * tests/test_s172_phase5_d6_threshold_path.py — the correction's own
        #     acceptance harness (Beta's nine threshold checks + the four residue
        #     session cases, with the three required threshold mutants and the
        #     residue mutant).
        # PWC/ZMQ/pwc_protocol remain unmodified, so coexistence still holds
        # (flagged for review).
        "miner/range_miner_protocol.py",
        "tests/test_s172_phase5_d6_threshold_path.py",
    }
    assert changed_py <= allowed, f"unexpected changed .py files: {changed_py - allowed}"
    # D3.25: PWC and ZMQ are deliberately no longer asserted unmodified — see the
    # registration note above. The PROTOCOL file stays frozen: D3.25 changes what
    # the two backends RETURN, never how they frame a message, so a pwc_protocol
    # edit would still be out-of-scope drift.
    for other in ("persistent/pwc_protocol.py",):
        assert other not in changed_py, f"{other} must remain unmodified (coexistence)"

    # 2) PWC + ZMQ backends still import with their trial entrypoints (coexist)
    import persistent_worker_coordinator as PWC
    import zmq_sqlite_coordinator as ZMQ
    assert hasattr(PWC, "run_trial_persistent") and hasattr(ZMQ, "run_trial_zmq_sqlite")

    # 3) the miner path: run_trial_miner builds + drives a real coordinator (arg
    #    plumbing -> CoordinatorConfig; a trial is created). Inject _serve to
    #    capture without needing a live worker fleet.
    def _capture(coord, ctx):
        return {"config": coord.config, "run_id": ctx["run_id"],
                "trial": coord.ledger.get_trial(ctx["run_id"]),
                "is_coordinator": isinstance(coord, RangeMinerCoordinator)}

    with tempfile.TemporaryDirectory() as tmp:
        out = run_trial_miner(
            "run-xyz", None, 7, "java_lcg", [1, 2, 3], 60, 0.25, 0.25, False,
            "/data/ds.json",
            seed_cap_nvidia=5_000_000, seed_cap_amd=2_000_000,
            seed_cap_nvidia_hybrid=2_500_000, seed_cap_amd_hybrid=1_000_000,
            staging_dir=os.path.join(tmp, "stg"),
            staging_high_water_bytes=123, staging_high_water_files=9,
            compute_lease_timeout=42.0, staging_timeout=99.0, _serve=_capture)
    cfg = out["config"]
    assert out["is_coordinator"] is True
    assert cfg.seed_cap_nvidia_hybrid == 2_500_000 and cfg.seed_cap_amd_hybrid == 1_000_000
    assert cfg.staging_high_water_bytes == 123 and cfg.staging_high_water_files == 9
    assert cfg.compute_lease_timeout == 42.0 and cfg.staging_timeout == 99.0
    assert out["trial"] is not None and out["trial"]["state"] == "running"


def gate23_non_regression():
    """Phase 0/1/2/3 harnesses still green as subprocesses — INCLUDING the
    Stage-0-patched Phase-3 resolver (dataset_sha256 gates 15-17)."""
    for rel in ("tests/test_prng_encoding.py",
                "tests/test_s172_phase1_scaffolding.py",
                "tests/test_s172_phase2_protocol.py",
                "tests/test_s172_phase3_worker.py"):
        r = subprocess.run(
            [sys.executable, rel], cwd=_ROOT,
            env={**os.environ, "PYTHONPATH": _ROOT},
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=600)
        assert r.returncode == 0, (
            f"{rel} exited {r.returncode}\n"
            f"stdout: {r.stdout[-800:]}\nstderr: {r.stderr[-400:]}")


# ---------------------------------------------------------------------------
# Gate 37 (Beta-required) — the REAL default serve_trial path over framed sockets
# ---------------------------------------------------------------------------
class _FakeWorker:
    """Minimal framed-socket worker: register → receive assign → respond. Speaks
    the REAL MinerFramedSocket wire (no shortcut). Behavior 'fail' sends a
    retryable stripe_error; 'complete' sends one inline sub-stripe covering the
    whole stripe + StripeComplete."""

    def __init__(self, host, port, hostname, gpu_id, behavior):
        self.host, self.port = host, port
        self.hostname, self.gpu_id = hostname, gpu_id
        self.worker_id = f"{hostname}:gpu{gpu_id}"
        self.behavior = behavior
        self.assigns_received = []
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
                    self.assigns_received.append((msg.stripe_id, msg.attempt))
                    self._respond(msg)
                elif msg.message_type == "shutdown":
                    break
        except Exception:
            self.err = traceback.format_exc()

    def _respond(self, assign):
        if self.behavior == "fail":
            self.fs.send_msg(StripeErrorMessage(
                worker_id=self.worker_id, stripe_id=assign.stripe_id, sub_index=0,
                error="synthetic retryable failure", retryable=True))
            return
        survivors = [[int(assign.seed_start), 0.9, None, [1]]]
        payload_obj, pb = build_substripe_payload_bytes(
            assign.stripe_id, 0, assign.seed_start, assign.seed_count, survivors)
        # [S172 D6] Echo the assignment's resolved threshold as the EFFECTIVE
        # value, exactly as the real worker does off its executor. The parent's
        # fail-closed provenance gate requires it on every D6-generated
        # assignment; a stub that omitted it would be modelling a worker that
        # violates the contract.
        eff = (assign.payload or {}).get("min_match_threshold")
        self.fs.send_msg(SubStripeResultMessage(
            worker_id=self.worker_id, stripe_id=assign.stripe_id, sub_index=0,
            seed_start=assign.seed_start, seed_count=assign.seed_count,
            survivor_count=1, inline=payload_obj, size_bytes=len(pb),
            sha256=hashlib.sha256(pb).hexdigest(), effective_threshold=eff))
        self.fs.send_msg(StripeCompleteMessage(
            worker_id=self.worker_id, stripe_id=assign.stripe_id,
            substripes_done=1, survivors_total=1, effective_threshold=eff))

    def stop(self):
        self._stop.set()
        try:
            self.fs.close()
        except Exception:
            pass


def gate37_serve_path_two_workers():
    """Beta-required: call run_trial_miner() with NO _serve, so the REAL default
    RangeMinerCoordinator.serve_trial() drives two loopback workers over real
    framed sockets. Proves: (1) both register; (2) stripes assigned; (3) a result
    traverses the real server path + is staged/verified/published; (4) the trial
    reaches a terminal state; (5) run_trial_miner returns a real dict; (6) NO
    NotImplementedError. Plus hybrid reassignment to a DIFFERENT worker."""
    with tempfile.TemporaryDirectory() as tmp:
        ds = os.path.join(tmp, "dataset.json")
        with open(ds, "w") as f:
            f.write('[{"draw":1},{"draw":2},{"draw":3}]')
        sink = _StubSink()

        # pre-bind an ephemeral listening socket (deterministic port for the gate)
        lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        lsock.bind(("127.0.0.1", 0))
        lsock.listen(8)
        port = lsock.getsockname()[1]

        holder = {}

        def run():
            try:
                holder["result"] = run_trial_miner(
                    "run-serve", None, 5, "java_lcg", [1, 2, 3], 30, 0.25, 0.25,
                    False, ds, worker_pool_size=2,
                    staging_dir=os.path.join(tmp, "stg"), phase5_sink=sink,
                    listen_sock=lsock, family_name="java_lcg_hybrid",
                    # D0 Blocker 2 (REV3): skip defaults are now fail-closed None at
                    # the entry point; a serve-path gate that legitimately runs with
                    # zero skip must pass skip_min/skip_max=0 explicitly.
                    skip_min=0, skip_max=0,
                    # D0 Blocker (REV4): window_size/offset are now fail-closed too;
                    # this serve-path gate must supply offset explicitly (window_size
                    # already passed).
                    offset=0,
                    workflow_phase=3, window_size=3, serve_timeout=20.0)
            except Exception:
                holder["err"] = traceback.format_exc()

        t = threading.Thread(target=run, daemon=True)
        t.start()

        # two fake workers over REAL framed sockets; w0 (hostA) registers first
        w0 = _FakeWorker("127.0.0.1", port, "hostA", 0, "fail")
        w1 = _FakeWorker("127.0.0.1", port, "hostB", 0, "complete")
        w0.connect_register()
        w1.connect_register()
        w0.start_loop()
        w1.start_loop()

        t.join(timeout=25)
        try:
            assert not t.is_alive(), "serve_trial did not terminate in time"
            assert "err" not in holder, holder.get("err")
            result = holder["result"]

            # (6) no NotImplementedError anywhere (workers + serve thread clean)
            assert w0.err is None and w1.err is None, (w0.err, w1.err)
            # (1) both workers registered over the real wire
            assert set(result["workers_registered"]) == {"hostA:gpu0", "hostB:gpu0"}
            # (2) stripes assigned — each fake worker received a real StripeAssignMessage
            assert w0.assigns_received and w1.assigns_received
            sid = w0.assigns_received[0][0]
            # hybrid reassignment: w0 got attempt 0 (failed), w1 got attempt 1
            assert any(a == 0 for (_, a) in w0.assigns_received)
            assert any(a == 1 for (_, a) in w1.assigns_received)
            st = result["stripes"][sid]
            assert st["claimed_by"] == "hostB:gpu0" and st["current_attempt"] == 1
            assert st["phase_degraded"] is True and st["state"] == "done"
            # (3) a result traversed the REAL serve path -> staged/verified/published
            assert len(result["manifests"]) >= 1
            m = result["manifests"][0]
            assert m["local_spool_path"] and os.path.isfile(m["local_spool_path"])
            assert len(sink.published) >= 1
            # (4) terminal state committed; (5) real result dict returned
            assert result["state"] == "committed" and result["committed"] is True
            assert len(sink.commits) == 1
        finally:
            w0.stop()
            w1.stop()
            try:
                lsock.close()
            except Exception:
                pass


# ===========================================================================
# CORRECTION 2 — Beta's six serve-path/ledger/wiring defect gates (38-47).
# Each FAILS on the pre-fix code (adversarial: two attempts sharing state,
# duplicate messages, cross-socket spoof, blocking transfers, terminal races,
# production-shaped consecutive trials).
# ===========================================================================
def gate38_stale_finish_private_path():
    """Defect 1: two attempts, same stripe/sub/seed-range (same sha). Attempt-0's
    stale _finalize_stage runs AFTER attempt 1 has staged its file. Because paths
    are attempt/generation-PRIVATE, attempt-1's file survives and attempt-0 removes
    only its OWN path. (OLD: shared path -> attempt 0 clobbers+deletes attempt 1.)"""
    with tempfile.TemporaryDirectory() as tmp:
        survivors = [[1, 0.9, 0, [1]]]
        coord = _coord_staging(tmp, miner_stripe_size=1000)
        connA = _register(coord, "hostA:gpu0")
        connB = _register(coord, "hostB:gpu0")
        sid = "run_sX"
        pb, size, sha = _canon(sid, 0, 0, 30, survivors)
        remote = f"/var/spool/miner/{sid}/0.json"
        coord.transfer = _StubTransfer(payloads={remote: pb})
        coord.ledger.add_stripe("run", sid, 0, 30, "java_lcg", 1, now=100.0)
        coord.ledger.claim_stripe("run", sid, "hostA:gpu0", 0, 1, lease_expires_at=150.0)
        connA.record_assignment(sid, 0)
        coord.ledger.record_substripe_result("run", sid, 0, 0, "hostA:gpu0", 0, 30, 1,
            remote_spool_path=remote, size_bytes=size, sha256=sha, now=110.0)
        # attempt-0 fetch begins (in flight): reservation held, temp fetched
        task0 = coord.begin_remote_stage(connA, "run", sid, 0, 0, remote, size, sha, now=110.0)
        # fence -> attempt 1 (gen++), staged by worker B (SAME range => SAME sha)
        coord.ledger.reclaim_expired_leases("run", now=1000.0)
        coord.ledger.claim_stripe("run", sid, "hostB:gpu0", 1, 1, lease_expires_at=1100.0)
        connB.record_assignment(sid, 1)
        coord.ledger.record_substripe_result("run", sid, 1, 0, "hostB:gpu0", 0, 30, 1,
            remote_spool_path=remote, size_bytes=size, sha256=sha, now=1001.0)
        res1 = coord.stage_remote_shard(connB, "run", sid, 1, 0, remote, size, sha, now=1001.0)
        assert res1["status"] == "verified"
        staged1 = res1["staged_path"]
        assert os.path.isfile(staged1)
        # attempt-0's stale finish
        res0 = coord.finish_remote_stage(task0, now=1002.0)
        assert res0["status"] == "stale"
        attempt1_file_after_stale_finish = os.path.isfile(staged1)
        assert attempt1_file_after_stale_finish is True    # Beta's exact probe
        sh1 = coord.ledger.get_shard("run", sid, 1, 0)
        assert sh1["local_staged_path"] == staged1 and os.path.isfile(sh1["local_staged_path"])
        assert task0.staged_path != staged1                # distinct private paths
        assert not os.path.isfile(task0.staged_path)       # stale removed only its own


def _drain_staging(coord):
    if coord._staging_executor is not None:
        coord._staging_executor.shutdown(wait=True)
        coord._staging_executor = None


def gate39_duplicate_result_one_reservation():
    """Defect 2: the SAME result delivered twice through _serve_dispatch yields
    exactly ONE held reservation / shard (duplicate dropped before staging).
    (OLD: return value ignored -> two reservations for one logical shard.)"""
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord_staging(tmp, miner_stripe_size=1000)
        conn = _register(coord, "hostA:gpu0")
        sid = "run_sD"
        coord.ledger.add_stripe("run", sid, 0, 30, "java_lcg", 1, now=100.0)
        coord.ledger.claim_stripe("run", sid, "hostA:gpu0", 0, 1, lease_expires_at=1e9)
        conn.record_assignment(sid, 0)
        wmap = {"hostA:gpu0": conn}
        elig = lambda: [conn]
        survivors = [[1, 0.9, None, [1]]]
        obj, pb = build_substripe_payload_bytes(sid, 0, 0, 30, survivors)
        msg = SubStripeResultMessage(
            worker_id="hostA:gpu0", stripe_id=sid, sub_index=0, seed_start=0,
            seed_count=30, survivor_count=1, inline=obj, size_bytes=len(pb),
            sha256=hashlib.sha256(pb).hexdigest())
        coord._serve_dispatch(msg, "run", "hostA:gpu0", wmap, elig)
        coord._serve_dispatch(msg, "run", "hostA:gpu0", wmap, elig)   # DUPLICATE
        _drain_staging(coord)
        eid = event_id_for("run", sid, 0, 0, 0)
        r = coord.ledger.get_reservation_by_event(eid)
        assert r is not None and r["status"] == "held"
        assert coord.reserved_files() == 1     # Beta probe: held_reservations_after_duplicate == 1
        assert len(coord.ledger.get_shards("run", sid, 0)) == 1


def gate40_dispatch_identity_spoof_rejected():
    """Defect 3: a frame physically received on worker A's socket but claiming
    worker B's id is rejected at dispatch (bound identity is authoritative), so it
    never reaches B's connection or mutates B's stripe. (OLD: connection resolved
    from msg.worker_id -> spoof passed L1 against B.)"""
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord_staging(tmp, miner_stripe_size=1000)
        connA = _register(coord, "hostA:gpu0")
        connB = _register(coord, "hostB:gpu0")
        sid = "run_sB"
        coord.ledger.add_stripe("run", sid, 0, 30, "java_lcg", 1, now=100.0)
        coord.ledger.claim_stripe("run", sid, "hostB:gpu0", 0, 1, lease_expires_at=1e9)
        connB.record_assignment(sid, 0)
        wmap = {"hostA:gpu0": connA, "hostB:gpu0": connB}
        elig = lambda: [connA, connB]
        obj, pb = build_substripe_payload_bytes(sid, 0, 0, 30, [[1, 0.9, None, [1]]])
        spoof = SubStripeResultMessage(
            worker_id="hostB:gpu0", stripe_id=sid, sub_index=0, seed_start=0,
            seed_count=30, survivor_count=1, inline=obj, size_bytes=len(pb),
            sha256=hashlib.sha256(pb).hexdigest())
        # REAL framed socket pair: send the spoof, read it on A's server side.
        a_srv, a_cli = socket.socketpair()
        srv, cli = MinerFramedSocket(a_srv), MinerFramedSocket(a_cli)
        try:
            cli.send_msg(spoof)
            received = srv.recv_msg()
            assert received.worker_id == "hostB:gpu0"
            # the receiving socket is BOUND to A -> the spoof is dropped
            coord._serve_dispatch(received, "run", "hostA:gpu0", wmap, elig)
            assert coord.ledger.get_shards("run", sid, 0) == []   # no ledger mutation
            assert coord.reserved_files() == 0                    # no reservation/stage
        finally:
            srv.close()
            cli.close()


def gate41_slow_fetch_nonblocking():
    """Defect 4a — dispatch-not-blocked. UPDATED FOR THE CORRECTION-6 SERIALIZED
    ADMISSION MODEL (flagged): the DEFECT this gate guards (a slow/blocked fetch must
    NOT stall the coordinator dispatch thread) is unchanged and still asserted — both
    `_serve_dispatch` calls return promptly while A's fetch is blocked. What CHANGED:
    under Approach A (serialize attempt-level staging), worker B's result is now
    DEFERRED (not staged in parallel) while attempt A is actively staging — the
    cross-attempt STAGING parallelism the old gate asserted is INTENTIONALLY traded
    away to guarantee no two-attempt circular wait (see gate 63). Dispatch liveness
    is preserved; only the parallel-staging expectation is corrected. When A completes
    + publishes (real lifecycle) and its capacity is released, B resumes and stages."""
    with tempfile.TemporaryDirectory() as tmp:
        gate = threading.Event()
        coord = _coord_staging(tmp, miner_stripe_size=1000)
        coord.ledger.set_trial_context("run", _ctx())
        connA = _register(coord, "hostA:gpu0")
        connB = _register(coord, "hostB:gpu0")
        sidA, sidB = "run_sA", "run_sB"
        _, pbA = build_substripe_payload_bytes(sidA, 0, 0, 30, [[1, 0.9, 0, [1]]])
        remoteA = f"/var/spool/miner/{sidA}/0.json"
        coord.transfer = _StubTransfer(payloads={remoteA: pbA}, fetch_gate=gate)
        for sid, wid, conn in ((sidA, "hostA:gpu0", connA), (sidB, "hostB:gpu0", connB)):
            coord.ledger.add_stripe("run", sid, 0, 30, "java_lcg", 1, now=100.0)
            coord.ledger.claim_stripe("run", sid, wid, 0, 1, lease_expires_at=1e9)
            conn.record_assignment(sid, 0)
        wmap = {"hostA:gpu0": connA, "hostB:gpu0": connB}
        elig = lambda: [connA, connB]
        shaA = hashlib.sha256(pbA).hexdigest()
        msgA = SubStripeResultMessage(worker_id="hostA:gpu0", stripe_id=sidA, sub_index=0,
            seed_start=0, seed_count=30, survivor_count=1, spool_path=remoteA,
            inline=None, size_bytes=len(pbA), sha256=shaA)
        objB, pbB = build_substripe_payload_bytes(sidB, 0, 0, 30, [[2, 0.8, None, [2]]])
        msgB = SubStripeResultMessage(worker_id="hostB:gpu0", stripe_id=sidB, sub_index=0,
            seed_start=0, seed_count=30, survivor_count=1, inline=objB,
            size_bytes=len(pbB), sha256=hashlib.sha256(pbB).hexdigest())
        # so A completes as soon as its (blocked) shard verifies -> releases the gate
        coord.ledger.record_stripe_complete("run", sidA, 0, "hostA:gpu0", 1, 1)
        try:
            # DEFECT ASSERTION (unchanged): dispatch is NOT blocked by the slow fetch.
            t0 = time.time()
            coord._serve_dispatch(msgA, "run", "hostA:gpu0", wmap, elig)   # A staging (blocked)
            coord._serve_dispatch(msgB, "run", "hostB:gpu0", wmap, elig)   # B deferred (serialized)
            assert time.time() - t0 < 1.0, "dispatch thread BLOCKED on the slow fetch"
            # C6 serialized behavior: while A stages, B is DEFERRED (bounded), NOT
            # staged in parallel and NOT partially consuming capacity.
            time.sleep(0.2)
            shB = coord.ledger.get_shard("run", sidB, 0, 0)
            assert shB is not None and shB["staging_status"] != SH_VERIFIED, \
                "B must be deferred (serialized), not staged in parallel with A"
            assert ("run", sidB, 0) in [(e[2], e[3], e[4]) for e in coord._deferred]
            shA = coord.ledger.get_shard("run", sidA, 0, 0)
            assert shA is not None and shA["staging_status"] != SH_VERIFIED  # fetch blocked
        finally:
            gate.set()      # release A's fetch -> A verifies, completes, publishes
        # A verifies -> completes -> publishes -> releases capacity -> B resumes and
        # stages (real lifecycle). Wait for B to VERIFY before draining, so B's
        # resubmit (from A's completion pump) is not racing executor shutdown.
        deadline = time.time() + 5
        while time.time() < deadline:
            shB = coord.ledger.get_shard("run", sidB, 0, 0)
            if shB and shB["staging_status"] == SH_VERIFIED:
                break
            time.sleep(0.02)
        _drain_staging(coord)
        # A completed + published; its capacity released, B resumed and staged.
        assert coord.ledger.get_shard("run", sidA, 0, 0)["staging_status"] == SH_VERIFIED
        assert coord.ledger.get_stripe("run", sidA)["state"] == ST_DONE
        assert coord.ledger.get_shard("run", sidB, 0, 0)["staging_status"] == SH_VERIFIED


def gate42_staging_timeout_matrix():
    """Defect 4b: a staging task exceeding staging_timeout is failed and routed
    through the phase-specific matrix (reassigned), file removed + reservation
    released (zero leak). (OLD: no timeout; failures never reached the matrix.)"""
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord_staging(tmp, miner_stripe_size=1000, staging_timeout=0.3)
        connA = _register(coord, "hostA:gpu0")
        connB = _register(coord, "hostB:gpu0")
        sid = "run_sT"
        _, pbA = build_substripe_payload_bytes(sid, 0, 0, 30, [[1, 0.9, 0, [1]]])
        remote = f"/var/spool/miner/{sid}/0.json"
        # slow fetch that never delivers (sleeps > timeout, then raises, no write)
        coord.transfer = _StubTransfer(payloads={remote: pbA}, fetch_delay=2.0,
                                       fetch_fail_after_delay=True)
        coord.ledger.add_stripe("run", sid, 0, 30, "java_lcg_hybrid", 3, now=100.0)
        coord.ledger.claim_stripe("run", sid, "hostA:gpu0", 0, 1, lease_expires_at=1e9)
        connA.record_assignment(sid, 0)
        wmap = {"hostA:gpu0": connA, "hostB:gpu0": connB}
        elig = lambda: [connA, connB]
        msg = SubStripeResultMessage(worker_id="hostA:gpu0", stripe_id=sid, sub_index=0,
            seed_start=0, seed_count=30, survivor_count=1, spool_path=remote,
            inline=None, size_bytes=len(pbA), sha256=hashlib.sha256(pbA).hexdigest())
        coord._serve_dispatch(msg, "run", "hostA:gpu0", wmap, elig)
        deadline = time.time() + 3
        while time.time() < deadline:
            st = coord.ledger.get_stripe("run", sid)
            if st["current_attempt"] == 1:
                break
            time.sleep(0.02)
        st = coord.ledger.get_stripe("run", sid)
        assert st["current_attempt"] == 1 and st["claimed_by"] == "hostB:gpu0"
        assert st["phase_degraded"] == 1        # timeout routed through the matrix
        assert coord.reserved_files() == 0      # reservation released (zero leak)


def gate43_admission_deferred_resume_real_lifecycle():
    """Defect 3 (C3) — REWRITTEN to use the REAL publish/ack lifecycle (the old
    gate 43 cheated: it manually acked a shard that was never published, which the
    real system can NEVER do — an ack only happens after finalize_stripe publishes
    the WHOLE attempt).

    Two 2-shard stripes (A, B) contend for a high-water that fits EXACTLY ONE full
    attempt (staging_high_water_files=2). The sub-stripes are delivered in the
    poison interleaving A.0, B.0, A.1, B.1.

    OLD (no admission): A.0 and B.0 both reserve (capacity full); A.1 and B.1 then
    back-pressure forever — NEITHER attempt can complete, so nothing is published,
    nothing can be acked, capacity never frees → self-deadlock (both stripes
    eventually time out / reassign). The gate's "both stripes done" never holds.

    FIXED (stripe-level admission + nonblocking deferred resume): attempt A is
    admitted (its whole 2-file footprint fits); attempt B is DEFERRED (admitting it
    would exceed high-water). A stages both sub-stripes, completes, and
    finalize_stripe PUBLISHES the whole attempt. ONLY THEN — via the REAL L6 ack
    path (ack_by_event_id on the PUBLISHED manifests) — does Phase 5 ack A's shards,
    which frees capacity; B is then admitted and its deferred sub-stripes resume and
    stage. No manual ack of an unpublished shard anywhere."""
    with tempfile.TemporaryDirectory() as tmp:
        sink = _StubSink()
        coord = _coord_staging(tmp, sink=sink, miner_stripe_size=1000,
                               staging_workers=2, staging_queue_depth=2,
                               staging_high_water_files=2,          # fits ONE attempt
                               staging_high_water_bytes=10 ** 12, staging_timeout=3.0)
        coord.ledger.set_trial_context("run", _ctx())
        coord.ledger.create_trial("run", 1, now=100.0)
        conn = _register(coord, "hostA:gpu0")
        elig = lambda: [conn]
        sidA, sidB = "run_sA", "run_sB"
        for sid in (sidA, sidB):
            coord.ledger.add_stripe("run", sid, 0, 30, "java_lcg", 1, now=100.0)
            coord.ledger.claim_stripe("run", sid, "hostA:gpu0", 0, 2, lease_expires_at=1e9)
            conn.record_assignment(sid, 0)

        def _mk(sid, sub, ss, sc):
            survs = [[sub, 0.9, None, [sub]]]
            obj, pb = build_substripe_payload_bytes(sid, sub, ss, sc, survs)
            return SubStripeResultMessage(
                worker_id="hostA:gpu0", stripe_id=sid, sub_index=sub, seed_start=ss,
                seed_count=sc, survivor_count=1, inline=obj, size_bytes=len(pb),
                sha256=hashlib.sha256(pb).hexdigest())

        subs = {(sid, sub): _mk(sid, sub, ss, sc)
                for sid in (sidA, sidB)
                for (sub, ss, sc) in [(0, 0, 15), (1, 15, 15)]}
        for (sid, sub), m in subs.items():
            coord.ledger.record_substripe_result(
                "run", sid, 0, sub, "hostA:gpu0", m.seed_start, m.seed_count, 1,
                size_bytes=m.size_bytes, sha256=m.sha256, remote_spool_path=None)

        # POISON interleaving A.0, B.0, A.1, B.1 (would deadlock without admission).
        futs = {}
        for (sid, sub) in [(sidA, 0), (sidB, 0), (sidA, 1), (sidB, 1)]:
            futs[(sid, sub)] = coord.enqueue_staging(
                "inline", conn, "run", sid, 0, sub, subs[(sid, sub)], elig)

        # attempt A's two sub-stripes both stage (admitted, footprint fits).
        futs[(sidA, 0)].result(timeout=5)
        futs[(sidA, 1)].result(timeout=5)
        for sub in (0, 1):
            assert coord.ledger.get_shard("run", sidA, 0, sub)["staging_status"] == SH_VERIFIED

        # attempt B is DEFERRED: its sub-stripes have not staged (admitting B would
        # exceed a high-water already committed to A).
        for sub in (0, 1):
            assert coord.ledger.get_shard("run", sidB, 0, sub)["staging_status"] != SH_VERIFIED

        # complete + finalize A -> the WHOLE attempt is PUBLISHED (real lifecycle).
        coord.ledger.record_stripe_complete("run", sidA, 0, "hostA:gpu0", 2, 2)
        coord.finalize_stripe("run", sidA, now=110.0, eligible_provider=elig)
        assert coord.ledger.get_stripe("run", sidA)["state"] == ST_DONE
        published_events = [m["event_id"] for m in sink.published
                            if m["stripe_id"] == sidA]
        assert len(published_events) == 2, "attempt A must publish BOTH shards first"

        # capacity is still held by A until Phase 5 acks; B remains unstaged.
        assert coord.reserved_files() == 2
        for sub in (0, 1):
            assert coord.ledger.get_shard("run", sidB, 0, sub)["staging_status"] != SH_VERIFIED

        # REAL L6 ack of A's PUBLISHED shards (no manual ack of an unpublished shard)
        # -> capacity frees -> deferred attempt B is admitted + resumes.
        for eid in published_events:
            assert coord.ack_by_event_id(eid, now=120.0) is True
        futs[(sidB, 0)].result(timeout=5)
        futs[(sidB, 1)].result(timeout=5)
        for sub in (0, 1):
            assert coord.ledger.get_shard("run", sidB, 0, sub)["staging_status"] == SH_VERIFIED

        coord.ledger.record_stripe_complete("run", sidB, 0, "hostA:gpu0", 2, 2)
        coord.finalize_stripe("run", sidB, now=130.0, eligible_provider=elig)
        assert coord.ledger.get_stripe("run", sidB)["state"] == ST_DONE
        # both stripes completed via admission + deferred resume; trial never failed.
        assert coord.ledger.get_trial("run")["state"] == "running"
        _drain_staging(coord)


def gate44_terminal_mutual_exclusion():
    """Defect 5a: committed and aborted are terminal + mutually exclusive. A
    committed trial can NEVER be flipped to aborted, and vice-versa. (OLD:
    mark_trial_aborted matched a committed trial and flipped it.)"""
    with tempfile.TemporaryDirectory() as tmp:
        sink = _StubSink()
        coord = _coord_staging(tmp, sink=sink)
        coord.ledger.create_trial("run", 1, now=100.0)
        coord.commit_trial("run", now=200.0)
        assert coord.ledger.get_trial("run")["state"] == "committed"
        r = coord.abort_trial("run", reason="x", now=201.0)
        assert r.get("refused") == "already_committed"
        assert coord.ledger.get_trial("run")["state"] == "committed"   # NOT flipped
        assert sink.aborts == []
    with tempfile.TemporaryDirectory() as tmp:
        sink = _StubSink()
        coord = _coord_staging(tmp, sink=sink)
        coord.ledger.create_trial("run", 1, now=100.0)
        coord.abort_trial("run", reason="x", now=200.0)
        assert coord.ledger.get_trial("run")["state"] == "aborted"
        try:
            coord.commit_trial("run", now=201.0)
        except TrialAborted:
            pass
        else:
            raise AssertionError("commit after abort must be refused")
        assert coord.ledger.get_trial("run")["state"] == "aborted"


def gate45_abort_runs_off_dispatch():
    """Defect 5b: a terminal abort triggered from the matrix runs OFF the dispatch
    thread (on the cleanup executor), never inline on the caller. (OLD: fail_trial
    called abort_trial synchronously on the receive/caller thread.)"""
    with tempfile.TemporaryDirectory() as tmp:
        sink = _StubSink()
        coord = _coord_staging(tmp, sink=sink)
        coord.ledger.create_trial("run", 1, now=100.0)
        conn = _register(coord, "hostA:gpu0")
        sid = "run_s0"
        coord.ledger.add_stripe("run", sid, 0, 30, "java_lcg", 1, now=100.0)
        coord.ledger.claim_stripe("run", sid, "hostA:gpu0", 0, 1, lease_expires_at=1e9)
        conn.record_assignment(sid, 0)
        caller = threading.current_thread().name
        act = coord.handle_stripe_failure("run", sid, retryable=False,
                                          eligible_workers=[conn], now=110.0)
        assert act["action"] == "fail_trial"
        assert coord.ledger.get_trial("run")["state"] == "aborted"
        assert sink.abort_threads, "abort must have been delivered"
        assert any("miner-cleanup" in n for n in sink.abort_threads), sink.abort_threads
        assert caller not in sink.abort_threads   # NOT inline on the dispatcher/caller


def gate46_commit_idempotent_by_event_id():
    """Defect 5c: a duplicate TrialCommit is a no-op by event_id — the sink is
    delivered exactly once. (OLD: no event_id; a second commit re-delivered.)"""
    with tempfile.TemporaryDirectory() as tmp:
        sink = _StubSink()
        coord = _coord_staging(tmp, sink=sink)
        coord.ledger.create_trial("run", 1, now=100.0)
        e1 = coord.commit_trial("run", now=200.0)
        e2 = coord.commit_trial("run", now=201.0)   # duplicate
        assert e1["event_id"] == e2["event_id"] == "run:commit"
        assert e2.get("duplicate") is True
        assert len(sink.commits) == 1               # delivered exactly once
        assert len(sink.commit_event_ids) == 1
        assert coord.ledger.get_trial("run")["commit_delivery_status"] == "done"


def gate47_production_call_shape_two_trials():
    """Defect 6: run_trial_miner called with the PRODUCTION arg shape (no test-only
    family/phase kwargs) for two consecutive trials -> distinct run_ids, NO
    stripe-ID/PK collision, and workflow phase / window params / staging_dir / bind
    address are the RESOLVED production values. (OLD: run_id = config filename ->
    PK collision on trial 2; phase 1 / window 1 / loopback defaults.)"""
    class _Cfg:
        window_size = 5
        sessions = ["evening", "midday"]
        offset = 3

    captured = []

    def _capture(coord, ctx):
        node = NodeConfig(hostname="h", spool_root="/var/spool/miner")
        conn = coord.register_worker(
            worker_id="hostA:gpu0", hostname="h", backend="cuda",
            capabilities={"seed_caps": dict(CAPS), "supported_variants": list(VARIANTS)},
            node_config=node)
        fam, ph = ctx["workflow_stages"][0]
        a = coord.assign_stripes(ctx["run_id"], fam, ph, 30, [conn],
                                 stripe_prefix=f"{ctx['run_id']}__st0")
        captured.append({
            "run_id": ctx["run_id"], "config": coord.config,
            "workflow_stages": ctx["workflow_stages"], "window_size": ctx["window_size"],
            "sessions": ctx["sessions"], "offset": ctx["offset"],
            "staging_dir": ctx["staging_dir"],
            "stripe_ids": [x["stripe_id"] for x in a],
        })
        return {"state": "captured", "run_id": ctx["run_id"]}

    with tempfile.TemporaryDirectory() as tmp:
        stg = os.path.join(tmp, "stg")
        for tno in (0, 1):
            run_trial_miner(
                coordinator_cfg="distributed_config.json", config=_Cfg(),
                trial_number=tno, prng_base="java_lcg", residues=[1, 2, 3],
                total_seeds=30, forward_threshold=0.25, reverse_threshold=0.25,
                test_both_modes=True, dataset_path="daily3.json", worker_pool_size=1,
                seed_cap_nvidia=5_000_000, seed_cap_amd=2_000_000,
                seed_cap_nvidia_hybrid=2_500_000, seed_cap_amd_hybrid=1_000_000,
                miner_stripe_size=67_108_864, miner_substripes=8,
                miner_output_dir=stg, staging_dir=None,
                staging_high_water_bytes=123, staging_high_water_files=9,
                compute_lease_timeout=42.0, staging_timeout=99.0,
                miner_host="0.0.0.0", miner_port=5700, node_allowlist=None,
                window_size=_Cfg.window_size, sessions=_Cfg.sessions, offset=_Cfg.offset,
                _serve=_capture)
        c0, c1 = captured
        assert c0["run_id"] != c1["run_id"]                       # distinct run_ids
        assert set(c0["stripe_ids"]).isdisjoint(c1["stripe_ids"])  # no PK collision
        assert c0["workflow_stages"] == workflow_stages_for("java_lcg", True)
        assert len(c0["workflow_stages"]) == 4                    # four families driven
        assert c0["window_size"] == 5 and c0["sessions"] == ["evening", "midday"]
        assert c0["offset"] == 3                                  # window params not dropped
        assert c0["config"].miner_host == "0.0.0.0"              # remote-reachable bind
        assert c0["config"].staging_high_water_bytes == 123
        assert c0["staging_dir"] == stg                           # defaulted from miner_output_dir


# ===========================================================================
# CORRECTION 3 — Beta's six async-staging + socket defect gates (48-54) plus the
# gate-43 rewrite above. Each FAILS on the pre-fix code (adversarial: a late/
# abandoned transfer write, an unbounded staging queue, a definitive
# reconciliation failure, a hybrid hash-mismatch, a silent/partial socket, and a
# 30s production abort).
# ===========================================================================
def gate48_late_transfer_no_orphan():
    """Defect 1 (C3): a fetch_remote that completes AFTER staging_timeout fired must
    NOT leave an orphan temp/staged file. The abandoned fetch's late write is
    tracked and removed; the reservation was already released at the timeout.
    (OLD: the daemon fetch thread was abandoned; its late write orphaned a
    .json.tmp.<pid> that nothing ever cleaned up.)"""
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord_staging(tmp, miner_stripe_size=1000)
        connA = _register(coord, "hostA:gpu0")
        sid = "run_sLATE"
        survivors = [[1, 0.9, 0, [1]]]
        pb, size, sha = _canon(sid, 0, 0, 10, survivors)
        remote = f"/var/spool/miner/{sid}/0.json"
        # slow fetch that WRITES the temp AFTER the timeout has fired (late success)
        coord.transfer = _StubTransfer(payloads={remote: pb}, fetch_delay=1.0)
        coord.ledger.add_stripe("run", sid, 0, 30, "java_lcg", 1, now=100.0)
        coord.ledger.claim_stripe("run", sid, "hostA:gpu0", 0, 1, lease_expires_at=1e9)
        connA.record_assignment(sid, 0)
        coord.ledger.record_substripe_result(
            "run", sid, 0, 0, "hostA:gpu0", 0, 10, 1,
            remote_spool_path=remote, size_bytes=size, sha256=sha, now=100.0)
        try:
            coord.begin_remote_stage(connA, "run", sid, 0, 0, remote, size, sha,
                                     now=100.0, fetch_timeout=0.3)
        except StagingTimeout:
            pass
        else:
            raise AssertionError("a fetch slower than fetch_timeout must raise StagingTimeout")
        # reservation released immediately at the timeout (no leak)
        assert coord.reserved_bytes() == 0 and coord.reserved_files() == 0
        # wait well past the late fetch's completion; the abandoned writer must have
        # removed its own late artifact — Beta's probe: NO files after late transfer.
        time.sleep(1.4)
        assert os.listdir(coord.config.staging_dir) == [], (
            f"late transfer left an orphan: {os.listdir(coord.config.staging_dir)}")
        assert coord.reserved_bytes() == 0 and coord.reserved_files() == 0


def gate49_dispatch_not_blocked_when_staging_saturated():
    """Defect 1a (C4) — REPLACES the old gate 49 (which validated the WRONG thing:
    it BLOCKED the producer and called it success, but the producer IS the dispatch
    thread — Beta's exact objection). The correct invariant: when every staging slot
    is saturated, the coordinator DISPATCH thread must still process another worker's
    message. Here a saturating sub_stripe_result is delivered on the dispatch thread,
    immediately followed by a DIFFERENT worker's heartbeat; the heartbeat MUST be
    processed (lease renewed), proving the result did not block dispatch.

    OLD (C3, blocking `_staging_slots().acquire()`): the saturating result blocks the
    dispatch thread on the full semaphore → the heartbeat is never reached
    (`third_dispatch_blocked=True`). FIXED: the result defers (nonblocking) → the
    heartbeat is processed on the same thread."""
    with tempfile.TemporaryDirectory() as tmp:
        gate = threading.Event()   # hold fetches so the two staging slots stay full
        coord = _coord_staging(tmp, miner_stripe_size=1000, staging_workers=2,
                               staging_queue_depth=0,          # semaphore == 2
                               staging_high_water_files=1000,
                               staging_high_water_bytes=4 * 1024 ** 3,
                               compute_lease_timeout=300.0)
        connA = _register(coord, "hostA:gpu0")
        connB = _register(coord, "hostB:gpu0")

        # SATURATE both staging slots with two blocking fetches (stripe X, 2 subs).
        sidX = "run_sX"
        coord.ledger.add_stripe("run", sidX, 0, 20, "java_lcg", 1, now=100.0)
        coord.ledger.claim_stripe("run", sidX, "hostA:gpu0", 0, 2, lease_expires_at=1e9)
        connA.record_assignment(sidX, 0)
        satpayloads = {}
        for i, ss in enumerate((0, 10)):
            _, pb = build_substripe_payload_bytes(sidX, i, ss, 10, [[i, 0.9, 0, [i]]])
            rp = f"/var/spool/miner/{sidX}/{i}.json"
            satpayloads[rp] = pb
        coord.transfer = _StubTransfer(payloads=satpayloads, fetch_gate=gate)
        for i, ss in enumerate((0, 10)):
            rp = f"/var/spool/miner/{sidX}/{i}.json"
            sha = hashlib.sha256(satpayloads[rp]).hexdigest()
            coord.ledger.record_substripe_result(
                "run", sidX, 0, i, "hostA:gpu0", ss, 10, 1, remote_spool_path=rp,
                size_bytes=len(satpayloads[rp]), sha256=sha, now=100.0)
            coord.enqueue_staging("remote", connA, "run", sidX, 0, i,
                                  SubStripeResultMessage(
                                      worker_id="hostA:gpu0", stripe_id=sidX, sub_index=i,
                                      seed_start=ss, seed_count=10, survivor_count=1,
                                      spool_path=rp, inline=None,
                                      size_bytes=len(satpayloads[rp]), sha256=sha),
                                  lambda: [connA, connB])
        # both slots now held by blocked fetches
        deadline = time.time() + 3
        while len(coord.transfer.fetch_calls) < 2 and time.time() < deadline:
            time.sleep(0.02)
        assert len(coord.transfer.fetch_calls) == 2, "staging slots not saturated"

        # stripe A (worker A) — the heartbeat target; stripe B (worker B) — a
        # saturating inline result that must NOT block the dispatch thread.
        for sid, wid, conn in (("run_sA", "hostA:gpu0", connA),
                               ("run_sB", "hostB:gpu0", connB)):
            coord.ledger.add_stripe("run", sid, 0, 30, "java_lcg", 1, now=100.0)
            coord.ledger.claim_stripe("run", sid, wid, 0, 1, lease_expires_at=1000.0)
            conn.record_assignment(sid, 0)
        wmap = {"hostA:gpu0": connA, "hostB:gpu0": connB}
        objB, pbB = build_substripe_payload_bytes("run_sB", 0, 0, 30, [[1, 0.9, None, [1]]])
        resultB = SubStripeResultMessage(
            worker_id="hostB:gpu0", stripe_id="run_sB", sub_index=0, seed_start=0,
            seed_count=30, survivor_count=1, inline=objB, size_bytes=len(pbB),
            sha256=hashlib.sha256(pbB).hexdigest())
        heartbeatA = MinerHeartbeatMessage(worker_id="hostA:gpu0", current_stripe_id="run_sA")

        done = threading.Event()

        def _dispatch_thread():
            # 1) a saturating result (admitted, but no slot) — MUST NOT block
            coord._serve_dispatch(resultB, "run", "hostB:gpu0", wmap, lambda: [connA, connB])
            # 2) a different worker's heartbeat — proves dispatch kept running
            coord._serve_dispatch(heartbeatA, "run", "hostA:gpu0", wmap, lambda: [connA, connB])
            done.set()

        lease_before = coord.ledger.get_stripe("run", "run_sA")["lease_expires_at"]
        t = threading.Thread(target=_dispatch_thread, daemon=True)
        t.start()
        try:
            assert done.wait(timeout=4), (
                "dispatch thread BLOCKED on saturated staging slots (Defect 1a)")
            lease_after = coord.ledger.get_stripe("run", "run_sA")["lease_expires_at"]
            assert lease_after > lease_before, "heartbeat not processed (dispatch stalled)"
            # the saturating result was deferred (nonblocking), not dropped/blocked
            assert coord.ledger.get_shard("run", "run_sB", 0, 0)["staging_status"] != SH_VERIFIED
        finally:
            gate.set()
            t.join(timeout=5)
            _drain_staging(coord)


def gate50_definitive_reconciliation_to_matrix():
    """Defect 4 (C3): a StripeComplete whose result set is present but does NOT
    reconcile (bad seed_count sum / coverage) is a DEFINITIVE failure routed through
    the phase-specific matrix EXACTLY ONCE — not parked in `staging` until the
    global trial timeout. A hybrid stripe reassigns to attempt 1 on a DIFFERENT
    worker. (OLD: finalize_stripe did nothing on a complete-but-invalid reconcile.)"""
    with tempfile.TemporaryDirectory() as tmp:
        sink = _StubSink()
        coord, (connA, connB) = _coord_workers(tmp, sink=sink, nvidia_hybrid=10)
        coord.ledger.create_trial("run", 1, now=100.0)
        a = coord.assign_stripes("run", "java_lcg_hybrid", 3, 30, [connA], now=100.0)[0]
        sid = a["stripe_id"]
        assert a["expected_substripes"] == 3
        # all THREE expected sub-stripes present + verified (substripes_match True)
        # but seed_count sum is 29 != 30 -> reconciliation DEFINITIVELY fails.
        for i, (ss, sc) in enumerate([(0, 10), (10, 10), (20, 9)]):
            coord.ledger.record_substripe_result("run", sid, 0, i, "hostA:gpu0", ss, sc, 1,
                                                  size_bytes=1, sha256="h", now=100.0)
            coord.ledger.mark_shard_verified("run", sid, 0, i, now=100.0)
        coord.ledger.record_stripe_complete("run", sid, 0, "hostA:gpu0", 3, 3)
        chk = coord.finalize_stripe("run", sid, now=110.0,
                                    eligible_provider=lambda: [connA, connB])
        assert chk.substripes_match and not chk.reconciled and not chk.is_complete
        # entered the matrix: reassigned to attempt 1 on a DIFFERENT worker (hybrid)
        st = coord.ledger.get_stripe("run", sid)
        assert st["current_attempt"] == 1 and st["claimed_by"] == "hostB:gpu0"
        assert st["phase_degraded"] == 1 and st["state"] == ST_CLAIMED
        assert coord.ledger.get_stripe("run", sid)["state"] != ST_STAGING
        assert coord.ledger.get_trial("run")["state"] == "running"
        # fired EXACTLY once: a second finalize on the now-claimed stripe is a no-op
        coord.finalize_stripe("run", sid, now=111.0,
                              eligible_provider=lambda: [connA, connB])
        assert coord.ledger.get_stripe("run", sid)["current_attempt"] == 1


def gate51_hash_mismatch_retryable_hybrid():
    """Defect 5 (C3): a hybrid-phase advertised-hash mismatch on attempt 0 feeds the
    one-retry path — reassignment to attempt 1 on a DIFFERENT worker
    (phase_degraded), NOT a trial abort. A constant-phase mismatch still fails
    closed. (OLD: the async handler marked every hash mismatch retryable=False,
    aborting a hybrid trial on attempt 0.)"""
    def _bad_inline_msg(sid, ss, sc):
        survs = [[1, 0.9, 0, [1]]]
        obj, pb = build_substripe_payload_bytes(sid, 0, ss, sc, survs)
        # advertise a WRONG sha/size so stage_inline_shard raises StagingHashMismatch
        return SubStripeResultMessage(
            worker_id="hostA:gpu0", stripe_id=sid, sub_index=0, seed_start=ss,
            seed_count=sc, survivor_count=1, inline=obj, size_bytes=len(pb) + 7,
            sha256="deadbeef" * 8)

    # (a) HYBRID: hash mismatch -> reassign to attempt 1 on a DIFFERENT worker
    with tempfile.TemporaryDirectory() as tmp:
        sink = _StubSink()
        coord, (connA, connB) = _coord_workers(tmp, sink=sink)
        coord.ledger.create_trial("run", 1, now=100.0)
        a = coord.assign_stripes("run", "java_lcg_hybrid", 3, 30, [connA], now=100.0)[0]
        sid = a["stripe_id"]
        m = _bad_inline_msg(sid, 0, 30)
        coord.ledger.record_substripe_result("run", sid, 0, 0, "hostA:gpu0", 0, 30, 1,
            size_bytes=m.size_bytes, sha256=m.sha256, remote_spool_path=None)
        f = coord.enqueue_staging("inline", connA, "run", sid, 0, 0, m,
                                  lambda: [connA, connB])
        f.result(timeout=5)
        st = coord.ledger.get_stripe("run", sid)
        assert st["current_attempt"] == 1 and st["claimed_by"] == "hostB:gpu0"
        assert st["phase_degraded"] == 1 and st["state"] == ST_CLAIMED
        assert coord.ledger.get_trial("run")["state"] == "running"   # NOT aborted
        _drain_staging(coord)

    # (b) CONSTANT: hash mismatch still fails closed (existing phase-1/2 behavior)
    with tempfile.TemporaryDirectory() as tmp:
        sink = _StubSink()
        coord, (connA, connB) = _coord_workers(tmp, sink=sink)
        coord.ledger.create_trial("run", 1, now=100.0)
        a = coord.assign_stripes("run", "java_lcg", 1, 30, [connA], now=100.0)[0]
        sid = a["stripe_id"]
        m = _bad_inline_msg(sid, 0, 30)
        coord.ledger.record_substripe_result("run", sid, 0, 0, "hostA:gpu0", 0, 30, 1,
            size_bytes=m.size_bytes, sha256=m.sha256, remote_spool_path=None)
        f = coord.enqueue_staging("inline", connA, "run", sid, 0, 0, m,
                                  lambda: [connA, connB])
        f.result(timeout=5)
        assert coord.ledger.get_trial("run")["state"] == "aborted"     # fail closed
        assert coord.ledger.get_stripe("run", sid)["current_attempt"] == 0
        _drain_staging(coord)


def _run_serve_thread(tmp, ds, sink, lsock, **kw):
    holder = {}

    def run():
        try:
            holder["result"] = run_trial_miner(
                "run-serve", None, 5, "java_lcg", [1, 2, 3], 30, 0.25, 0.25,
                False, ds, worker_pool_size=kw.pop("worker_pool_size", 1),
                staging_dir=os.path.join(tmp, "stg"), phase5_sink=sink,
                listen_sock=lsock, family_name="java_lcg_hybrid", workflow_phase=3,
                # D0 Blocker 2 (REV3): explicit zero skip on the serve-path gate.
                skip_min=0, skip_max=0,
                # D0 Blocker (REV4): explicit offset on the serve-path gate.
                offset=0,
                window_size=3, **kw)
        except Exception:
            holder["err"] = traceback.format_exc()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, holder


def gate52_silent_client_timeout_and_deadline():
    """Defect 6 (C3): a client that connects and sends NOTHING, before any valid
    worker, must NOT wedge the server — (a) the serve/timeout loop keeps running and
    fires the trial abort; (b) the silent connection is dropped on its read deadline
    (well before the serve timeout). (OLD: the loop did a blocking recv_msg right
    after accept, wedging registration/dispatch AND the timeout until the socket
    closed.)"""
    with tempfile.TemporaryDirectory() as tmp:
        ds = os.path.join(tmp, "dataset.json")
        with open(ds, "w") as f:
            f.write('[{"draw":1},{"draw":2},{"draw":3}]')
        sink = _StubSink()
        lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        lsock.bind(("127.0.0.1", 0))
        lsock.listen(8)
        port = lsock.getsockname()[1]

        t, holder = _run_serve_thread(tmp, ds, sink, lsock, worker_pool_size=1,
                                      serve_timeout=3.0, serve_read_deadline=1.0)
        # a silent client connects FIRST and sends nothing
        silent = socket.create_connection(("127.0.0.1", port))
        silent.settimeout(5.0)
        connected_at = time.time()
        try:
            # (b) the server drops the silent socket on the read deadline (~1s),
            # WELL before the 3s serve timeout — recv returns EOF.
            data = silent.recv(1)
            dropped_at = time.time()
            assert data == b"", "silent socket should be dropped (EOF), not fed data"
            assert dropped_at - connected_at < 3.0, "dropped only at serve timeout, not read deadline"
        finally:
            silent.close()

        t.join(timeout=15)
        assert not t.is_alive(), "serve_trial wedged on the silent client"
        assert "err" not in holder, holder.get("err")
        result = holder["result"]
        # (a) the timeout still fired -> trial aborted; no worker ever registered
        assert result["state"] == "aborted"
        assert result["workers_registered"] == []
        try:
            lsock.close()
        except Exception:
            pass


def gate53_partial_frame_nonblocking_and_dropped():
    """Defect 6 (C3): a client that sends a header + only PART of the body (never a
    complete frame), connecting before a valid worker, must NOT block the valid
    worker's registration/dispatch, and the stuck connection is eventually dropped.
    (OLD: the blocking recv_msg on a partial frame wedged the whole serve loop.)"""
    with tempfile.TemporaryDirectory() as tmp:
        ds = os.path.join(tmp, "dataset.json")
        with open(ds, "w") as f:
            f.write('[{"draw":1},{"draw":2},{"draw":3}]')
        sink = _StubSink()
        lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        lsock.bind(("127.0.0.1", 0))
        lsock.listen(8)
        port = lsock.getsockname()[1]

        t, holder = _run_serve_thread(tmp, ds, sink, lsock, worker_pool_size=1,
                                      serve_timeout=20.0, serve_read_deadline=1.5)
        # partial-frame client connects FIRST: a 4-byte header declaring 4096 body
        # bytes, then only 8 body bytes, then silence — never a complete frame.
        partial = socket.create_connection(("127.0.0.1", port))
        partial.settimeout(8.0)
        import struct as _struct
        partial.sendall(_struct.pack(">I", 4096) + b"12345678")
        time.sleep(0.4)   # let the server accept it + its reader block mid-frame

        # a VALID worker registers + completes AFTER the partial client — it must not
        # be blocked by the wedged partial connection.
        w = _FakeWorker("127.0.0.1", port, "hostA", 0, "complete")
        w.connect_register()
        w.start_loop()

        try:
            # (b) the partial connection is dropped (read deadline / shutdown) -> EOF
            data = partial.recv(1)
            assert data == b"", "partial-frame socket should be dropped (EOF)"
        finally:
            partial.close()

        t.join(timeout=25)
        assert not t.is_alive(), "serve_trial wedged on the partial-frame client"
        assert "err" not in holder, holder.get("err")
        assert w.err is None, w.err
        result = holder["result"]
        # (a) the valid worker registered + drove the trial to commit (NOT blocked)
        assert "hostA:gpu0" in result["workers_registered"]
        assert result["state"] == "committed" and result["committed"] is True
        w.stop()
        try:
            lsock.close()
        except Exception:
            pass


def gate54_production_serve_timeout_unbounded():
    """Production-timeout correction (Beta): run_trial_miner with NO serve_timeout
    imposes NO 30s abort (unbounded — a real scan runs far longer); a CONFIGURED
    timeout is still honored. (OLD: run_trial_miner defaulted serve_timeout to 30.0
    and the production integration never overrode it, aborting valid long scans.)"""
    # (a) no serve_timeout -> the resolved context carries None (unbounded)
    captured = {}

    def _capture(coord, ctx):
        captured["serve_timeout"] = ctx.get("serve_timeout", "MISSING")
        return {"state": "captured"}

    with tempfile.TemporaryDirectory() as tmp:
        run_trial_miner("run-p", None, 1, "java_lcg", [1, 2, 3], 30, 0.25, 0.25,
                        False, "/data/ds.json", staging_dir=os.path.join(tmp, "s"),
                        _serve=_capture)
    assert captured["serve_timeout"] is None, (
        f"production serve_timeout must default to None (unbounded), "
        f"got {captured['serve_timeout']!r}")

    # (b) a CONFIGURED timeout is honored by the REAL serve loop: with no worker it
    # aborts at ~1s (not never, not 30s).
    with tempfile.TemporaryDirectory() as tmp:
        ds = os.path.join(tmp, "dataset.json")
        with open(ds, "w") as f:
            f.write('[{"draw":1}]')
        sink = _StubSink()
        lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        lsock.bind(("127.0.0.1", 0))
        lsock.listen(8)
        t, holder = _run_serve_thread(tmp, ds, sink, lsock, worker_pool_size=1,
                                      serve_timeout=1.0, serve_read_deadline=5.0)
        started = time.time()
        t.join(timeout=15)
        elapsed = time.time() - started
        assert not t.is_alive(), "configured serve_timeout was not honored"
        assert "err" not in holder, holder.get("err")
        assert holder["result"]["state"] == "aborted"
        assert elapsed < 8.0, f"configured 1s timeout not honored (took {elapsed:.1f}s)"
        try:
            lsock.close()
        except Exception:
            pass


# ===========================================================================
# CORRECTION 4 — Beta's four overload/heterogeneous-worker defect gates (55-59)
# plus the gate-49 REPLACEMENT above. Each FAILS on the C3 code.
# ===========================================================================
def gate55_oversized_attempt_fails_fast():
    """Defect 1b (C4): an attempt whose WHOLE footprint cannot fit the configured
    high-water (files OR bytes) FAILS FAST as a capacity/config error — it is NOT
    admitted to wait forever. (OLD: `_try_admit_locked` always admitted the first
    attempt; a 2-substripe stripe under staging_high_water_files=1 then parked in
    `staging` forever — second shard could never reserve, nothing published/acked.)"""
    with tempfile.TemporaryDirectory() as tmp:
        sink = _StubSink()
        coord = _coord_staging(tmp, sink=sink, miner_stripe_size=1000,
                               staging_high_water_files=1,
                               staging_high_water_bytes=10 ** 12)
        coord.ledger.create_trial("run", 1, now=100.0)
        conn = _register(coord, "hostA:gpu0")
        sid = "run_sBIG"
        coord.ledger.add_stripe("run", sid, 0, 30, "java_lcg", 1, now=100.0)
        # expected_substripes = 2 > staging_high_water_files = 1 -> can NEVER fit
        coord.ledger.claim_stripe("run", sid, "hostA:gpu0", 0, 2, lease_expires_at=1e9)
        conn.record_assignment(sid, 0)
        assert coord._attempt_exceeds_highwater(
            coord.ledger.get_stripe("run", sid)) is not None
        obj, pb = build_substripe_payload_bytes(sid, 0, 0, 15, [[1, 0.9, None, [1]]])
        m = SubStripeResultMessage(worker_id="hostA:gpu0", stripe_id=sid, sub_index=0,
            seed_start=0, seed_count=15, survivor_count=1, inline=obj,
            size_bytes=len(pb), sha256=hashlib.sha256(pb).hexdigest())
        coord.ledger.record_substripe_result("run", sid, 0, 0, "hostA:gpu0", 0, 15, 1,
            size_bytes=m.size_bytes, sha256=m.sha256, remote_spool_path=None)
        coord.enqueue_staging("inline", conn, "run", sid, 0, 0, m, lambda: [conn]).result(timeout=5)
        # failed fast: trial explicitly failed, NOT parked in staging, nothing
        # admitted or deferred waiting forever.
        assert coord.ledger.get_trial("run")["state"] == "aborted"
        assert coord.ledger.get_stripe("run", sid)["state"] != ST_STAGING
        assert coord._admitted == {} and coord._deferred == []
        _drain_staging(coord)


def gate56_bounded_deferred_queue():
    """Defect 1c (C4): the deferred queue is bounded by COUNT (staging_deferred_max)
    and retained BYTES (staging_high_water_bytes). Beyond the bound, dispatch
    back-pressures via the matrix instead of retaining more payloads. (OLD:
    `_deferred` was a plain unbounded list — 100 un-admittable attempts sat with
    `deferred_len=100` against a bound of 2, each retaining its inline payload.)"""
    with tempfile.TemporaryDirectory() as tmp:
        gate = threading.Event()
        sink = _StubSink()
        coord, (connA, connB) = _coord_workers(
            tmp, sink=sink, nvidia_hybrid=10, staging_workers=2, staging_queue_depth=0,
            staging_high_water_files=1, staging_high_water_bytes=10 ** 12,
            staging_deferred_max=2)
        coord.ledger.create_trial("run", 1, now=100.0)
        # capacity holder: one admitted remote attempt whose fetch blocks, holding
        # the single high-water file AND keeping admission committed.
        sidH = "run_sH"
        _, pbH = build_substripe_payload_bytes(sidH, 0, 0, 10, [[0, 0.9, 0, [0]]])
        remoteH = f"/var/spool/miner/{sidH}/0.json"
        shaH = hashlib.sha256(pbH).hexdigest()
        coord.transfer = _StubTransfer(payloads={remoteH: pbH}, fetch_gate=gate)
        coord.ledger.add_stripe("run", sidH, 0, 30, "java_lcg_hybrid", 3, now=100.0)
        coord.ledger.claim_stripe("run", sidH, "hostA:gpu0", 0, 1, lease_expires_at=1e9)
        connA.record_assignment(sidH, 0)
        coord.ledger.record_substripe_result("run", sidH, 0, 0, "hostA:gpu0", 0, 10, 1,
            remote_spool_path=remoteH, size_bytes=len(pbH), sha256=shaH)
        coord.enqueue_staging("remote", connA, "run", sidH, 0, 0,
            SubStripeResultMessage(worker_id="hostA:gpu0", stripe_id=sidH, sub_index=0,
                seed_start=0, seed_count=10, survivor_count=1, spool_path=remoteH,
                inline=None, size_bytes=len(pbH), sha256=shaH), lambda: [connA, connB])
        deadline = time.time() + 3
        while not coord.transfer.fetch_calls and time.time() < deadline:
            time.sleep(0.02)
        assert coord.reserved_files() == 1

        # submit far more un-admittable single-sub HYBRID attempts than the bound.
        N = 12
        for k in range(N):
            sid = f"run_sD{k}"
            coord.ledger.add_stripe("run", sid, 0, 30, "java_lcg_hybrid", 3, now=100.0)
            coord.ledger.claim_stripe("run", sid, "hostA:gpu0", 0, 1, lease_expires_at=1e9)
            connA.record_assignment(sid, 0)
            obj, pb = build_substripe_payload_bytes(sid, 0, 0, 30, [[1, 0.9, None, [1]]])
            m = SubStripeResultMessage(worker_id="hostA:gpu0", stripe_id=sid, sub_index=0,
                seed_start=0, seed_count=30, survivor_count=1, inline=obj,
                size_bytes=len(pb), sha256=hashlib.sha256(pb).hexdigest())
            coord.ledger.record_substripe_result("run", sid, 0, 0, "hostA:gpu0", 0, 30, 1,
                size_bytes=m.size_bytes, sha256=m.sha256, remote_spool_path=None)
            coord.enqueue_staging("inline", connA, "run", sid, 0, 0, m, lambda: [connA, connB])
        # D1c: the deferred queue NEVER exceeds its configured count/byte bounds.
        assert len(coord._deferred) <= 2, f"deferred_len={len(coord._deferred)} > bound 2"
        assert coord._deferred_retained_bytes() <= coord.config.staging_high_water_bytes
        # excess was back-pressured via the matrix (hybrid reassign) — trial runs on.
        assert coord.ledger.get_trial("run")["state"] == "running"
        gate.set()
        _drain_staging(coord)


def gate57_variant_filtered_scheduling():
    """Defect 2 (C4): the scheduling pool is filtered to variant-compatible workers
    BEFORE round-robin; a family with NO compatible worker fails the trial
    explicitly. (OLD: round-robin `workers[i % len]` THEN a variant check left the
    mismatched stripe `pending` with no later path — stranded forever.)"""
    # (a) mixed pool: A=pcg32-only, B=java_lcg -> BOTH java_lcg stripes go to B
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord_staging(tmp, miner_stripe_size=30)
        A = _register(coord, "hostA:gpu0", variants=["pcg32", "pcg32_hybrid"])
        B = _register(coord, "hostB:gpu0", variants=VARIANTS)
        assigns = coord.assign_stripes("run", "java_lcg", 1, 60, [A, B], now=100.0)
        assert len(assigns) == 2
        for a in assigns:
            assert a["claimed"] is True and a["worker_id"] == "hostB:gpu0", a
            assert coord.ledger.get_stripe("run", a["stripe_id"])["claimed_by"] == "hostB:gpu0"
            assert coord.ledger.get_stripe("run", a["stripe_id"])["state"] == ST_CLAIMED
        # none stranded pending
        assert coord.ledger.stripes_by_state("run", ST_PENDING) == []

    # (b) a family NO worker supports -> the trial FAILS EXPLICITLY, not hangs
    with tempfile.TemporaryDirectory() as tmp:
        ds = os.path.join(tmp, "dataset.json")
        with open(ds, "w") as f:
            f.write('[{"draw":1},{"draw":2},{"draw":3}]')
        sink = _StubSink()
        lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        lsock.bind(("127.0.0.1", 0))
        lsock.listen(8)
        port = lsock.getsockname()[1]
        holder = {}

        def run():
            try:
                holder["result"] = run_trial_miner(
                    "run-nv", None, 5, "java_lcg", [1, 2, 3], 30, 0.25, 0.25, False, ds,
                    worker_pool_size=1, staging_dir=os.path.join(tmp, "stg"),
                    phase5_sink=sink, listen_sock=lsock,
                    family_name="mt19937", workflow_phase=1,   # NO worker supports it
                    # D0 Blocker 2 (REV3): explicit zero skip on the serve-path gate.
                    skip_min=0, skip_max=0,
                    # D0 Blocker (REV4): explicit offset on the serve-path gate.
                    offset=0,
                    window_size=3, serve_timeout=20.0, serve_read_deadline=10.0)
            except Exception:
                holder["err"] = traceback.format_exc()

        t = threading.Thread(target=run, daemon=True)
        t.start()
        w = _FakeWorker("127.0.0.1", port, "hostA", 0, "complete")
        w.connect_register()
        w.start_loop()
        t.join(timeout=15)
        try:
            assert not t.is_alive(), "trial hung on a family with no compatible worker"
            assert "err" not in holder, holder.get("err")
            assert holder["result"]["state"] == "aborted"   # failed explicitly
        finally:
            w.stop()
            try:
                lsock.close()
            except Exception:
                pass


def gate58_one_socket_per_worker():
    """Defect 3 (C4): a socket registers exactly once and one worker_id maps to at
    most one live socket. (OLD: `_serve_register` accepted a second REGISTER on a
    bound socket -> rebind + stale mapping `[A,B]`; a second socket could also claim
    an already-connected worker_id -> two sockets share one identity.)"""
    def _reg(wid, host):
        return RegisterMessage(
            worker_id=wid, hostname=host, gpu_id=0, gpu_name="x", backend="cuda",
            vram_bytes=12 * 1024 ** 3,
            capabilities={"seed_caps": dict(CAPS), "supported_variants": list(VARIANTS)})

    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord_staging(tmp)
        s1a, _s1b = socket.socketpair()
        s2a, _s2b = socket.socketpair()
        try:
            fs_by_sock = {s1a: MinerFramedSocket(s1a), s2a: MinerFramedSocket(s2a)}
            worker_by_sock, wconn_by_worker, fs_by_worker, registered = {}, {}, {}, []

            # first register on socket 1 as worker A
            st = coord._serve_register(_reg("A:gpu0", "A"), s1a, None, fs_by_sock,
                                       worker_by_sock, wconn_by_worker, fs_by_worker,
                                       registered)
            assert st == "ok"
            assert worker_by_sock[s1a] == "A:gpu0" and set(fs_by_worker) == {"A:gpu0"}

            # (a) a SECOND register on the SAME socket, now claiming B -> rejected;
            # the socket STAYS bound to A, no [A,B].
            st2 = coord._serve_register(_reg("B:gpu0", "B"), s1a, None, fs_by_sock,
                                        worker_by_sock, wconn_by_worker, fs_by_worker,
                                        registered)
            assert st2 == "reject_rebind"
            assert worker_by_sock[s1a] == "A:gpu0"          # stays bound to A
            assert set(fs_by_worker) == {"A:gpu0"}          # NO [A,B]
            assert "B:gpu0" not in wconn_by_worker

            # (b) a SECOND socket registering the already-connected worker_id A ->
            # rejected; exactly one live socket remains for A (the original).
            st3 = coord._serve_register(_reg("A:gpu0", "A"), s2a, None, fs_by_sock,
                                        worker_by_sock, wconn_by_worker, fs_by_worker,
                                        registered)
            assert st3 == "reject_dup_worker"
            assert s2a not in worker_by_sock                # s2 NOT bound
            assert fs_by_worker["A:gpu0"] is fs_by_sock[s1a]  # still the ORIGINAL
        finally:
            for s in (s1a, _s1b, s2a, _s2b):
                try:
                    s.close()
                except Exception:
                    pass


def gate59_orphan_fetch_threads_live_bound():
    """Defect 2 (C5) — REPLACES the C4 gate 59, which checked only the REGISTRY
    LENGTH, not the real threads. The C4 fix started the fetch thread BEFORE the cap
    check, so a refused fetch still leaked a LIVE thread (Beta: cap 2, seven hung
    fetches → 2 registry entries but SEVEN live `miner-fetch` threads). This gate
    asserts the REAL resource — LIVE `miner-fetch` threads counted via
    threading.enumerate() — stays ≤ cap, and every excess job fails with a capacity
    error because its thread is NEVER launched once the budget is exhausted."""
    def _live_fetch_threads():
        return [t for t in threading.enumerate()
                if t.name == "miner-fetch" and t.is_alive()]

    with tempfile.TemporaryDirectory() as tmp:
        gate = threading.Event()          # NEVER set -> every fetch blocks (hung)
        cap, N = 2, 7
        baseline = len(_live_fetch_threads())
        coord = _coord_staging(tmp, staging_orphan_fetch_max=cap)
        coord.transfer = _StubTransfer(payloads={}, fetch_gate=gate)
        node = NodeConfig(hostname="h", spool_root=SPOOL_ROOT)
        timeouts, cap_errors = 0, 0
        try:
            for k in range(N):            # N=7 > cap=2 permanently-blocked fetches
                try:
                    coord._fetch_with_timeout(
                        node, f"/var/spool/miner/x{k}.json",
                        os.path.join(tmp, f"t{k}.tmp"), 0.2)
                except StagingTimeout:
                    timeouts += 1
                except StagingError as e:
                    cap_errors += 1
                    assert "capacity" in str(e).lower(), str(e)
            # Beta's EXACT probe: LIVE miner-fetch threads (NOT registry length) ≤ cap.
            live_now = len(_live_fetch_threads())
            assert live_now <= baseline + cap, (
                f"{live_now - baseline} live miner-fetch threads exceeds cap {cap} "
                "(threads leaked past the registry)")
            # the excess jobs failed with a capacity error; only `cap` threads launched
            assert timeouts == cap and cap_errors == N - cap
        finally:
            gate.set()   # release the blocked fetch threads
            coord.account_orphan_fetches(join_timeout=0.5)


def gate60_large_remote_spool_fails_fast():
    """Defect 1 (C5): a REMOTE shard whose ADVERTISED size_bytes exceeds the byte
    high-water can never fit — it must fail IMMEDIATELY (capacity error), not be
    admitted on a stale 48 MiB/file estimate and then loop forever on
    StagingBackPressure. Asserts the REAL resource (advertised bytes), not the
    estimate. (OLD: `_attempt_footprint` used INLINE_BYTE_LIMIT, so a 70 MiB spool
    passed a 60 MiB high-water then dead-looped.)"""
    MB = 1024 * 1024
    with tempfile.TemporaryDirectory() as tmp:
        sink = _StubSink()
        coord = _coord_staging(tmp, sink=sink, miner_stripe_size=1000,
                               staging_high_water_files=100,
                               staging_high_water_bytes=60 * MB)
        coord.transfer = _StubTransfer(payloads={})   # must never even be fetched
        coord.ledger.create_trial("run", 1, now=100.0)
        conn = _register(coord, "hostA:gpu0")
        sid = "run_sBIGSPOOL"
        coord.ledger.add_stripe("run", sid, 0, 30, "java_lcg", 1, now=100.0)
        coord.ledger.claim_stripe("run", sid, "hostA:gpu0", 0, 1, lease_expires_at=1e9)
        conn.record_assignment(sid, 0)
        remote = f"/var/spool/miner/{sid}/0.json"
        big = 70 * MB                                  # advertised REMOTE size > HW
        m = SubStripeResultMessage(worker_id="hostA:gpu0", stripe_id=sid, sub_index=0,
            seed_start=0, seed_count=30, survivor_count=1, spool_path=remote,
            inline=None, size_bytes=big, sha256="deadbeef")
        coord.ledger.record_substripe_result("run", sid, 0, 0, "hostA:gpu0", 0, 30, 1,
            remote_spool_path=remote, size_bytes=big, sha256="deadbeef")
        # the FILES-only footprint guard does NOT catch it (1 file ≤ 100) — only the
        # ACTUAL advertised-byte guard does.
        assert coord._attempt_exceeds_highwater(coord.ledger.get_stripe("run", sid)) is None
        coord.enqueue_staging("remote", conn, "run", sid, 0, 0, m, lambda: [conn]).result(timeout=5)
        # failed IMMEDIATELY: trial failed, stripe left the waiting state, nothing
        # admitted/deferred, and the oversized spool was NEVER fetched (no dead-loop).
        assert coord.ledger.get_trial("run")["state"] == "aborted"
        assert coord.ledger.get_stripe("run", sid)["state"] not in (ST_CLAIMED, ST_STAGING)
        assert coord._admitted == {} and coord._deferred == []
        assert coord.transfer.fetch_calls == []
        _drain_staging(coord)


def gate61_disconnected_worker_not_eligible():
    """Defect 3 (C5): a worker whose socket is dropped is EVICTED from EVERY structure
    the eligible pool is built from (wconn_by_worker / self.connections / registered),
    so it is never handed NEW stripes. (OLD: `_drop_conn` cleared only fs_by_*, so
    `_eligible()` still included the dead worker → stripes claimed by an unreachable
    A, unsendable, stuck until lease expiry.)"""
    def _reg(wid, host):
        return RegisterMessage(worker_id=wid, hostname=host, gpu_id=0, gpu_name="x",
            backend="cuda", vram_bytes=12 * 1024 ** 3,
            capabilities={"seed_caps": dict(CAPS), "supported_variants": list(VARIANTS)})

    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord_staging(tmp, miner_stripe_size=30)
        sA, _sAb = socket.socketpair()
        sB, _sBb = socket.socketpair()
        try:
            fs_by_sock = {sA: MinerFramedSocket(sA), sB: MinerFramedSocket(sB)}
            worker_by_sock, wconn_by_worker, fs_by_worker, registered = {}, {}, {}, []
            for sk, wid, host in ((sA, "A:gpu0", "A"), (sB, "B:gpu0", "B")):
                assert coord._serve_register(
                    _reg(wid, host), sk, None, fs_by_sock, worker_by_sock,
                    wconn_by_worker, fs_by_worker, registered) == "ok"
            assert set(wconn_by_worker) == {"A:gpu0", "B:gpu0"}

            # A disconnects BEFORE assignment
            coord._drop_conn(sA, fs_by_sock, worker_by_sock, fs_by_worker,
                             wconn_by_worker, registered)
            # A evicted EVERYWHERE the eligible pool is built from
            assert "A:gpu0" not in wconn_by_worker
            assert "A:gpu0" not in coord.connections
            assert "A:gpu0" not in registered
            assert "A:gpu0" not in fs_by_worker
            eligible = [w for w in wconn_by_worker.values() if not w.quarantined]
            assert [w.worker_id for w in eligible] == ["B:gpu0"]

            # every new compatible stripe now goes to B, NONE to A, none unsendable
            assigns = coord.assign_stripes("run", "java_lcg", 1, 60, eligible, now=100.0)
            assert len(assigns) == 2
            for a in assigns:
                assert a["worker_id"] == "B:gpu0" and a["claimed"] is True
            claimedA = [s for s in coord.ledger.all_stripes("run")
                        if s["claimed_by"] == "A:gpu0"]
            assert claimedA == [], "a stripe was left claimed by the disconnected A"
        finally:
            for s in (sA, _sAb, sB, _sBb):
                try:
                    s.close()
                except Exception:
                    pass


# ===========================================================================
# CORRECTION 6 — Beta's two mandatory admission gates (62-63): ONE coherent
# byte model (serialized attempt-level staging, actual advertised sizes).
# ===========================================================================
def gate62_tiny_inline_admission():
    """Defect (C6): a TINY inline attempt must not be permanently deferred by a
    static per-file estimate. 2 expected sub-stripes, 60 MiB byte high-water,
    ~100-byte payloads → the attempt STAGES and COMPLETES. (OLD C5: `_attempt_footprint`
    budgeted 2 × 48 MiB = 96 MiB > 60 MiB → `_try_admit_locked` deferred it FOREVER;
    no capacity event can ever make a static 96 MiB fit 60 MiB.)"""
    MB = 1024 * 1024
    with tempfile.TemporaryDirectory() as tmp:
        sink = _StubSink()
        coord = _coord_staging(tmp, sink=sink, miner_stripe_size=1000,
                               staging_high_water_files=2,
                               staging_high_water_bytes=60 * MB)
        coord.ledger.set_trial_context("run", _ctx())
        coord.ledger.create_trial("run", 1, now=100.0)
        conn = _register(coord, "hostA:gpu0")
        sid = "run_sTINY"
        coord.ledger.add_stripe("run", sid, 0, 30, "java_lcg", 1, now=100.0)
        coord.ledger.claim_stripe("run", sid, "hostA:gpu0", 0, 2, lease_expires_at=1e9)  # E=2
        conn.record_assignment(sid, 0)
        futs = []
        for sub, ss, sc in ((0, 0, 15), (1, 15, 15)):
            obj, pb = build_substripe_payload_bytes(sid, sub, ss, sc, [[sub, 0.9, None, [sub]]])
            assert len(pb) < 4096, "payload must be tiny (far below the 60 MiB high-water)"
            m = SubStripeResultMessage(worker_id="hostA:gpu0", stripe_id=sid, sub_index=sub,
                seed_start=ss, seed_count=sc, survivor_count=1, inline=obj,
                size_bytes=len(pb), sha256=hashlib.sha256(pb).hexdigest())
            coord.ledger.record_substripe_result("run", sid, 0, sub, "hostA:gpu0", ss, sc, 1,
                size_bytes=m.size_bytes, sha256=m.sha256, remote_spool_path=None)
            futs.append(coord.enqueue_staging("inline", conn, "run", sid, 0, sub, m, lambda: [conn]))
        # both tiny shards STAGE (NOT permanently deferred by a static 96 MiB estimate)
        for f in futs:
            f.result(timeout=5)
        for sub in (0, 1):
            assert coord.ledger.get_shard("run", sid, 0, sub)["staging_status"] == SH_VERIFIED
        assert coord._deferred == [], "tiny attempt must not sit in the deferred queue"
        # complete + PUBLISH the whole attempt (real lifecycle)
        coord.ledger.record_stripe_complete("run", sid, 0, "hostA:gpu0", 2, 2)
        coord.finalize_stripe("run", sid, now=110.0)
        assert coord.ledger.get_stripe("run", sid)["state"] == ST_DONE
        assert len(sink.published) == 2
        _drain_staging(coord)


def gate63_cross_attempt_remote_serialized():
    """Defect (C6): two remote attempts must NOT partial-admit into a circular wait.
    Two attempts × two 70 MiB remote shards, 200 MiB global high-water → ONE attempt
    makes forward progress and PUBLISHES (via the REAL publish path); the other WAITS
    without partially consuming capacity; no circular wait. (OLD C5: the static 96 MiB
    estimate admitted BOTH; A.0 + B.0 reserve 140 MiB, each second shard needs +70 MiB
    (210 > 200) → both wait forever, neither publishes, published = 0.)"""
    MB = 1024 * 1024
    with tempfile.TemporaryDirectory() as tmp:
        sink = _StubSink()
        coord = _coord_staging(tmp, sink=sink, miner_stripe_size=1000,
                               staging_workers=2, staging_queue_depth=2,
                               staging_high_water_files=100,
                               staging_high_water_bytes=200 * MB, staging_timeout=30.0)
        coord.ledger.set_trial_context("run", _ctx())
        coord.ledger.create_trial("run", 1, now=100.0)
        conn = _register(coord, "hostA:gpu0")
        # identical 70 MiB payload shared by all four shards (one object, ~73 MB RAM)
        pb = b"\0" * (70 * MB)
        size, sha = len(pb), hashlib.sha256(pb).hexdigest()
        sidA, sidB = "run_sA", "run_sB"
        payloads, msgs = {}, {}
        for sid in (sidA, sidB):
            coord.ledger.add_stripe("run", sid, 0, 30, "java_lcg", 1, now=100.0)
            coord.ledger.claim_stripe("run", sid, "hostA:gpu0", 0, 2, lease_expires_at=1e9)  # E=2
            conn.record_assignment(sid, 0)
            for sub, ss, sc in ((0, 0, 15), (1, 15, 15)):
                remote = f"/var/spool/miner/{sid}/{sub}.json"
                payloads[remote] = pb
                msgs[(sid, sub)] = SubStripeResultMessage(
                    worker_id="hostA:gpu0", stripe_id=sid, sub_index=sub, seed_start=ss,
                    seed_count=sc, survivor_count=1, spool_path=remote, inline=None,
                    size_bytes=size, sha256=sha)
                coord.ledger.record_substripe_result("run", sid, 0, sub, "hostA:gpu0", ss, sc, 1,
                    remote_spool_path=remote, size_bytes=size, sha256=sha)
        coord.transfer = _StubTransfer(payloads=payloads)
        elig = lambda: [conn]
        # POISON interleaving A.0, B.0, A.1, B.1 (the C5 circular-wait order).
        fA0 = coord.enqueue_staging("remote", conn, "run", sidA, 0, 0, msgs[(sidA, 0)], elig)
        coord.enqueue_staging("remote", conn, "run", sidB, 0, 0, msgs[(sidB, 0)], elig)
        fA1 = coord.enqueue_staging("remote", conn, "run", sidA, 0, 1, msgs[(sidA, 1)], elig)
        coord.enqueue_staging("remote", conn, "run", sidB, 0, 1, msgs[(sidB, 1)], elig)
        try:
            # attempt A makes forward progress: BOTH its shards stage (serialized,
            # 140 MiB ≤ 200 MiB). On C5 A.1 back-pressure-waits forever -> TimeoutError.
            fA0.result(timeout=25)
            fA1.result(timeout=25)
            for sub in (0, 1):
                assert coord.ledger.get_shard("run", sidA, 0, sub)["staging_status"] == SH_VERIFIED
            # A completes + PUBLISHES the whole attempt (real lifecycle)
            coord.ledger.record_stripe_complete("run", sidA, 0, "hostA:gpu0", 2, 2)
            coord.finalize_stripe("run", sidA, now=110.0)
            assert coord.ledger.get_stripe("run", sidA)["state"] == ST_DONE
            assert len([m for m in sink.published if m["stripe_id"] == sidA]) == 2
            # B WAITED without partially consuming capacity: neither B shard verified,
            # neither B shard holds a reservation; only A's 2 files are held.
            for sub in (0, 1):
                assert coord.ledger.get_shard("run", sidB, 0, sub)["staging_status"] != SH_VERIFIED
                assert coord.ledger.get_reservation_by_event(
                    event_id_for("run", sidB, 0, sub, 0)) is None
            assert coord.reserved_files() == 2   # only A's shards held (no B partial)
        finally:
            # cancel active stripes first so any (regression-induced) back-pressure-
            # waiting job bails instead of hanging the drain — the FAIL on broken code
            # is then a clean assertion/timeout, never a wedged harness.
            coord.ledger.cancel_active_stripes("run")
            _drain_staging(coord)


def main():
    print("\nS172 Phase 4 coordinator acceptance harness — 36 gates + serve-path (37)")
    print("=" * 70)
    _check("Gate 1: macro-stripe partition + assign",            gate1_macro_partition_assign)
    _check("Gate 2: multiple sub-stripe results (shard ledger)", gate2_multiple_substripe_results)
    _check("Gate 3: missing/duplicate/overlapping sub_index",    gate3_bad_sub_index_not_done)
    _check("Gate 4: shard-level done conditions",                gate4_done_conditions)
    _check("Gate 5: staging state; reclaim skips staging",       gate5_staging_state_reclaim)
    _check("Gate 6: StripeComplete waits for staging verify",    gate6_wait_for_staging)
    _check("Gate 7: partial-attempt cleanup before retry (B2)",  gate7_partial_attempt_cleanup)
    _check("Gate 8: TrialCommit vs TrialAbort (B2)",             gate8_commit_vs_abort)
    _check("Gate 9: constant-phase immediate fail (B3)",         gate9_constant_phase_immediate_fail)
    _check("Gate 10: hybrid one-retry-then-fail (B3)",           gate10_hybrid_one_retry_then_fail)
    _check("Gate 11: retryable=False immediate fail (B3)",       gate11_non_retryable_immediate_fail)
    _check("Gate 12: lease expiry phase-specific policy (B3)",   gate12_lease_expiry_policy)
    _check("Gate 13: inline result normalized (B4)",             gate13_inline_normalized)
    _check("Gate 14: remote staging happy path (Decision B)",    gate14_remote_happy_path)
    _check("Gate 15: hash mismatch → no delete, retry path",     gate15_hash_mismatch)
    _check("Gate 16: byte/file reservation high-water",          gate16_high_water_mark)
    _check("Gate 17: spool path restricted to spool root",       gate17_spool_root_restriction)
    _check("Gate 18: connection-bound identity (Decision A)",    gate18_connection_bound_identity)
    _check("Gate 19: cap/variant mismatch at registration",      gate19_cap_or_variant_mismatch)
    _check("Gate 20: resolver end-to-end, wrong dataset_sha256",  gate20_resolver_end_to_end)
    _check("Gate 21: no Phase-5 assembly in coordinator (§3.A)",  gate21_no_phase5_assembly)
    _check("Gate 22: coexistence (use_range_miner, PWC/ZMQ)",     gate22_coexistence)
    _check("Gate 23: Phase 0/1/2/3 non-regression subprocess",    gate23_non_regression)
    _check("Gate 24: stale-attempt fencing (L1)",                gate24_stale_attempt_fencing)
    _check("Gate 25: enqueue does not release capacity (L2)",    gate25_enqueue_does_not_release)
    _check("Gate 26: ack + local delete releases capacity (L2)", gate26_ack_releases_capacity)
    _check("Gate 27: high-water counts unacked staged (L2)",     gate27_high_water_counts_unacked)
    _check("Gate 28: whole-trial abort + L7 retain (L3/L7)",     gate28_whole_trial_abort)
    _check("Gate 29: four-cap validation + quarantine (L4)",     gate29_four_cap_validation)
    _check("Gate 30: staging resources injectable config (L4)",  gate30_staging_resources_configurable)
    _check("Gate 31: durable remote-delete status (SC1)",        gate31_durable_remote_delete)
    _check("Gate 32: stale async-task fencing (L5)",             gate32_stale_async_task_fencing)
    _check("Gate 33: event-id ack idempotency (L6)",             gate33_event_id_ack_idempotency)
    _check("Gate 34: abort-discard race, sync return (L7)",      gate34_abort_discard_race)
    _check("Gate 35: full completion reconciliation (L8)",       gate35_full_reconciliation)
    _check("Gate 36: failure-path reservation cleanup (L8)",     gate36_failure_path_cleanup)
    _check("Gate 37: REAL serve_trial path, two workers (Beta)", gate37_serve_path_two_workers)
    _check("Gate 38: [D1] stale finish uses private path",       gate38_stale_finish_private_path)
    _check("Gate 39: [D2] duplicate result → one reservation",   gate39_duplicate_result_one_reservation)
    _check("Gate 40: [D3] cross-socket identity spoof rejected",  gate40_dispatch_identity_spoof_rejected)
    _check("Gate 41: [D4a] slow fetch does not block dispatch",   gate41_slow_fetch_nonblocking)
    _check("Gate 42: [D4b] staging timeout → matrix, zero leak",  gate42_staging_timeout_matrix)
    _check("Gate 43: [C3-D3] admission + deferred resume (real lifecycle)", gate43_admission_deferred_resume_real_lifecycle)
    _check("Gate 44: [D5a] terminal states mutually exclusive",   gate44_terminal_mutual_exclusion)
    _check("Gate 45: [D5b] abort runs off the dispatch thread",   gate45_abort_runs_off_dispatch)
    _check("Gate 46: [D5c] TrialCommit idempotent by event_id",   gate46_commit_idempotent_by_event_id)
    _check("Gate 47: [D6] production call shape, two trials",     gate47_production_call_shape_two_trials)
    _check("Gate 48: [C3-D1] late transfer leaves no orphan",     gate48_late_transfer_no_orphan)
    _check("Gate 49: [C4-D1a] dispatch not blocked when saturated", gate49_dispatch_not_blocked_when_staging_saturated)
    _check("Gate 50: [C3-D4] definitive reconcile → matrix",      gate50_definitive_reconciliation_to_matrix)
    _check("Gate 51: [C3-D5] hash mismatch retryable (hybrid)",   gate51_hash_mismatch_retryable_hybrid)
    _check("Gate 52: [C3-D6] silent client → timeout + deadline", gate52_silent_client_timeout_and_deadline)
    _check("Gate 53: [C3-D6] partial frame → nonblocking + drop", gate53_partial_frame_nonblocking_and_dropped)
    _check("Gate 54: [C3-prod] serve_timeout unbounded default",  gate54_production_serve_timeout_unbounded)
    _check("Gate 55: [C4-D1b] oversized attempt fails fast",       gate55_oversized_attempt_fails_fast)
    _check("Gate 56: [C4-D1c] bounded deferred queue",            gate56_bounded_deferred_queue)
    _check("Gate 57: [C4-D2] variant-filtered scheduling",        gate57_variant_filtered_scheduling)
    _check("Gate 58: [C4-D3] one socket per worker_id",           gate58_one_socket_per_worker)
    _check("Gate 59: [C5-D2] LIVE orphan fetch threads ≤ cap",     gate59_orphan_fetch_threads_live_bound)
    _check("Gate 60: [C5-D1] large remote spool fails fast",       gate60_large_remote_spool_fails_fast)
    _check("Gate 61: [C5-D3] disconnected worker not eligible",    gate61_disconnected_worker_not_eligible)
    _check("Gate 62: [C6] tiny-inline attempt stages+completes",   gate62_tiny_inline_admission)
    _check("Gate 63: [C6] two-remote-attempt serialized, no wait", gate63_cross_attempt_remote_serialized)
    print("=" * 70)

    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    # 54 _check calls total = 36 brief Phase-4 gates + gate 37 (Beta serve-path) +
    # gates 38-47 (Correction-2 six-defect gates) + gate 43 rewrite (Correction-3
    # Defect 3, real lifecycle) + gates 48-54 (Correction-3 async-staging/socket/
    # production-timeout defect gates); gate 23 additionally runs the Phase 0/1/2/3
    # subprocess non-regression.
    print(f"\n{passed}/{total} checks green "
          "(36 brief + gate 37 + 38-47 C2 + 48-54 C3 + 49/55-58 C4 + 59-61 C5 "
          "+ gate 41 updated + 62-63 C6 defect gates; gate 23 = subprocess non-regression)")
    if passed != total:
        print("\nFAILURES (DO NOT COMMIT):")
        for name, ok, tb in _results:
            if not ok:
                print(f"\n--- {name} ---\n{tb}")
        sys.exit(1)
    print("\nAll checks green — S172 Phase 4 coordinator (incl. the real serve_trial "
          "path) is contract-validated (pending Team Alpha + Team Beta review).")
    sys.exit(0)


if __name__ == "__main__":
    main()
