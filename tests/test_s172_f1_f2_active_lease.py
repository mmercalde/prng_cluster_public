#!/usr/bin/env python3
"""S172 F1/F2 — ACTIVE-LEASE SCHEDULER + TERMINAL OBSERVABILITY.

Beta ruling "GATE-12 F1 FORENSICS / LEASE AMENDMENT" (2026-08-09): production
defect CONFIRMED, F1/F2 amendment AUTHORIZED, gate-12 rerun REMAINS HELD.

THE DEFECT THIS SUITE EXISTS FOR (`distributed_config_t1_689f3cd9`, 2026-08-09):
`assign_stripes` claimed every stripe of a stage in ONE loop with ONE `now`,
stamping each `now + compute_lease_timeout`. Workers execute serially, so at 4
stripes/worker the fourth began with ~40-70 s of its 300 s lease left. Three
workers that were ACTIVELY STREAMING RESULTS (last shards 12:47:11.3 / .12.1 /
.12.6) had leases that expired at 12:47:05.487, and the constant-phase policy
correctly failed the trial on that bad input. The matrix was right; its input
was wrong.

EVERY GATE HERE IS DETERMINISTIC. Time is passed in as `now`, never slept —
`process_lease_expiry`, `schedule_pending_stripes` and `_renew_active_lease` all
take an explicit clock, so "delivery slower than the stage-wide lease" is
expressed as arithmetic rather than as a race the CI host can lose.

Run:  source ~/venvs/torch/bin/activate
      python3 -u tests/test_s172_f1_f2_active_lease.py | tee /tmp/f1f2.log
"""
import logging
import os
import sys
import tempfile
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


from miner.range_miner_coordinator import (  # noqa: E402
    ST_CANCELLED,
    ST_CLAIMED,
    ST_DONE,
    ST_PENDING,
    ST_STAGING,
    CoordinatorConfig,
    LeaseInvariantError,
    MinerLedger,
    NodeConfig,
    RangeMinerCoordinator,
    TC_COMPUTE_LEASE_EXPIRY,
    TC_STRIPE_ERROR,
    TerminalRecord,
    expected_substripes_for,
)
import miner.range_miner_coordinator as COORD  # noqa: E402
from miner.range_miner_protocol import (  # noqa: E402
    MinerHeartbeatMessage,
)

LEASE = 300.0
MACRO = 1000                      # small macro so N stripes is easy to state
CAPS = {"amd": 2_000_000, "nvidia": 5_000_000,
        "amd_hybrid": 1_000_000, "nvidia_hybrid": 2_500_000}
VARIANTS = ["java_lcg", "java_lcg_reverse", "java_lcg_hybrid",
            "java_lcg_hybrid_reverse"]
SPOOL_ROOT = "/var/spool/miner"


def _coord(tmp, **cfg):
    cfg.setdefault("miner_stripe_size", MACRO)
    cfg.setdefault("compute_lease_timeout", LEASE)
    cfg.setdefault("staging_dir", os.path.join(tmp, "staging"))
    ledger = MinerLedger(os.path.join(tmp, "l.db"))
    return RangeMinerCoordinator(CoordinatorConfig(**cfg), ledger)


def _register(coord, wid, backend="cuda", variants=None, now=100.0):
    node = NodeConfig(hostname=wid.split(":")[0], spool_root=SPOOL_ROOT,
                      ssh_address="10.0.0.9", ssh_user="michael")
    return coord.register_worker(
        worker_id=wid, hostname=wid.split(":")[0], backend=backend,
        capabilities={"seed_caps": dict(CAPS),
                      "supported_variants": list(variants or VARIANTS)},
        node_config=node, now=now)


def _stage(coord, n_workers, n_stripes, fam="java_lcg", phase=1, now=100.0,
           tmp_workers=None):
    """Create a stage of `n_stripes` over `n_workers` and return (conns, assigns)."""
    conns = tmp_workers or [_register(coord, f"host{i}:gpu0")
                            for i in range(n_workers)]
    assigns = coord.assign_stripes(
        "run", fam, phase, MACRO * n_stripes, conns,
        stripe_prefix="run__st0", now=now)
    return conns, assigns


def _mutant_red(fn, label):
    """VIR-2 positive control: `fn` must RAISE. A mutation that the gate cannot
    detect is a vacuous gate, so the gate proves its own detection power."""
    try:
        fn()
    except Exception:                                        # noqa: BLE001
        return
    raise AssertionError(
        f"MUTANT SURVIVED ({label}) — the gate did not detect it, so it is "
        f"vacuous and proves nothing")


class _LogCapture:
    """Capture ERROR records from the coordinator's own logger (F2 surface 3)."""

    def __init__(self):
        self.records = []
        self._h = None

    def __enter__(self):
        outer = self

        class _H(logging.Handler):
            def emit(self, record):
                if record.levelno >= logging.ERROR:
                    outer.records.append(record.getMessage())

        self._h = _H()
        # Attach to the coordinator module's OWN logger object. Using a dotted
        # name here would be a guess: the production log shows this logger
        # emitting as `range_miner_coordinator`, not `miner.range_miner_coordinator`,
        # so a name-based handler silently captures nothing and the gate would pass
        # vacuously by finding no records to contradict it.
        COORD.logger.addHandler(self._h)
        return self

    def __exit__(self, *a):
        COORD.logger.removeHandler(self._h)
        return False


# ===========================================================================
# G-F1-QUEUE-NO-LEASE
# ===========================================================================
def g_queue_no_lease():
    """N > W: exactly W enter compute-active `claimed`; the remaining N-W stay
    `pending` with claimed_by = NULL AND lease_expires_at = NULL.

    The NULL lease is the whole point. On 2026-08-09 the backlog carried a lease
    that was already ticking, which is how a stripe that had not started could be
    24 seconds from expiry."""
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp)
        conns, assigns = _stage(coord, n_workers=3, n_stripes=8, now=1000.0)
        assert len(assigns) == 8
        claimed = [a for a in assigns if a["claimed"]]
        queued = [a for a in assigns if not a["claimed"]]
        assert len(claimed) == 3, f"expected W=3 claimed, got {len(claimed)}"
        assert len(queued) == 5, f"expected N-W=5 queued, got {len(queued)}"
        # every planned row EXISTS — the governed geometry is not truncated
        assert len(coord.ledger.all_stripes("run")) == 8
        for a in claimed:
            st = coord.ledger.get_stripe("run", a["stripe_id"])
            assert st["state"] == ST_CLAIMED
            assert st["claimed_by"] is not None
            assert st["lease_expires_at"] == 1000.0 + LEASE, st
        for a in queued:
            st = coord.ledger.get_stripe("run", a["stripe_id"])
            assert st["state"] == ST_PENDING, st
            assert st["claimed_by"] is None, f"backlog carries a claimer: {st}"
            assert st["lease_expires_at"] is None, f"BACKLOG HAS A LEASE: {st}"
        # one stripe per worker, and all three workers used
        assert sorted(a["worker_id"] for a in claimed) == \
            sorted(c.worker_id for c in conns)


# ===========================================================================
# G-F1-ONE-ACTIVE   (+ the bulk-claim mutant)
# ===========================================================================
def g_one_active():
    """No serial worker holds more than one compute-active claim, and a mutation
    restoring BULK CLAIM reds this gate."""
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp)
        conns, assigns = _stage(coord, n_workers=2, n_stripes=6, now=1000.0)
        holders = [s["claimed_by"] for s in coord.ledger.all_stripes("run")
                   if s["state"] == ST_CLAIMED]
        assert len(holders) == len(set(holders)) == 2, holders
        assert coord.ledger.compute_busy_worker_ids("run") == set(holders)

        # MUTANT: the pre-amendment behaviour — claim every planned stripe to a
        # worker in one pass, exactly as `assign_stripes` did before F1.
        def _bulk_claim_mutant():
            pend = coord.ledger.pending_stripes("run", "run__st0")
            for row in pend:
                coord.ledger.claim_stripe(
                    "run", row["stripe_id"], conns[0].worker_id, 0, 1,
                    1000.0 + LEASE)
            holders2 = [s["claimed_by"] for s in coord.ledger.all_stripes("run")
                        if s["state"] == ST_CLAIMED]
            assert len(holders2) == len(set(holders2)), (
                "a worker holds more than one compute-active claim")
        _mutant_red(_bulk_claim_mutant, "bulk claim restored")


# ===========================================================================
# G-F1-FRESH-HANDOFF
# ===========================================================================
def g_fresh_handoff():
    """When worker A completes X and receives pending Y, Y's lease is
    handoff_time + timeout — NOT stage-creation time + timeout.

    This is the arithmetic the incident turned on: under bulk claim Y's deadline
    was fixed the moment the STAGE was planned, so Y's usable lease shrank by
    however long X took."""
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp)
        STAGE_T = 1000.0
        conns, assigns = _stage(coord, n_workers=1, n_stripes=2, now=STAGE_T)
        A = conns[0]
        x = next(a for a in assigns if a["claimed"])
        y = next(a for a in assigns if not a["claimed"])
        stx = coord.ledger.get_stripe("run", x["stripe_id"])
        assert stx["lease_expires_at"] == STAGE_T + LEASE

        # X's compute finishes 250 s later (inside its own lease, as it should be)
        HANDOFF_T = STAGE_T + 250.0
        coord.ledger.record_stripe_complete(
            "run", x["stripe_id"], 0, A.worker_id, 1, 0)
        assert coord.ledger.get_stripe("run", x["stripe_id"])["state"] == ST_STAGING
        assert coord.ledger.compute_busy_worker_ids("run") == set(), \
            "a STAGING stripe must not hold the worker's compute slot"

        placed = coord.schedule_pending_stripes(
            "run", "java_lcg", 1, conns, stage_prefix="run__st0", now=HANDOFF_T)
        assert len(placed) == 1 and placed[0]["stripe_id"] == y["stripe_id"]
        sty = coord.ledger.get_stripe("run", y["stripe_id"])
        assert sty["state"] == ST_CLAIMED and sty["claimed_by"] == A.worker_id
        assert sty["lease_expires_at"] == HANDOFF_T + LEASE, (
            f"Y's lease is {sty['lease_expires_at']}, expected fresh "
            f"{HANDOFF_T + LEASE}; a stage-creation stamp would give "
            f"{STAGE_T + LEASE}")
        # the defect, stated as arithmetic: under the old stamp Y would have had
        # only 50 s of lease left at the instant it started.
        assert (STAGE_T + LEASE) - HANDOFF_T == 50.0


# ===========================================================================
# G-F1-PROGRESS-RENEWAL  /  G-F1-HEARTBEAT-RENEWAL
# ===========================================================================
def _one_claimed(coord, tmp_now=1000.0):
    conns, assigns = _stage(coord, n_workers=1, n_stripes=1, now=tmp_now)
    a = assigns[0]
    assert a["claimed"]
    return conns[0], a["stripe_id"]


def g_progress_renewal():
    """A valid accepted SubStripeResultMessage for the ACTIVE attempt extends its
    lease — driven through the production `_renew_active_lease` predicate."""
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp)
        A, sid = _one_claimed(coord)
        before = coord.ledger.get_stripe("run", sid)["lease_expires_at"]
        assert before == 1000.0 + LEASE
        ok = coord._renew_active_lease(A, "run", sid, A.worker_id,
                                       source="sub_stripe_result", now=1250.0)
        assert ok is True
        after = coord.ledger.get_stripe("run", sid)["lease_expires_at"]
        assert after == 1250.0 + LEASE > before, (before, after)


def g_heartbeat_renewal():
    """The pre-existing heartbeat renewal still works, through the SAME predicate,
    and through the real `_serve_dispatch` heartbeat branch (not just the helper)."""
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp)
        A, sid = _one_claimed(coord)
        before = coord.ledger.get_stripe("run", sid)["lease_expires_at"]
        hb = MinerHeartbeatMessage(worker_id=A.worker_id, current_stripe_id=sid)
        coord._serve_dispatch(hb, "run", A.worker_id, {A.worker_id: A},
                              lambda: [A])
        after = coord.ledger.get_stripe("run", sid)["lease_expires_at"]
        assert after > before, (
            f"heartbeat did not renew via the real dispatch path: {before} -> {after}")


# ===========================================================================
# G-F1-SCOPE-RENEWAL — every forbidden case, each with the branch that rejects it
# ===========================================================================
def g_scope_renewal():
    """NONE of these renew. Beta calls the distinction load-bearing:
    progress on THIS active attempt renews THIS active attempt — not
    'any traffic from the host keeps everything it owns alive'."""
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp)
        A = _register(coord, "hostA:gpu0")
        B = _register(coord, "hostB:gpu0")
        assigns = coord.assign_stripes("run", "java_lcg", 1, MACRO * 3, [A, B],
                                       stripe_prefix="run__st0", now=1000.0)
        claimed = [a for a in assigns if a["claimed"]]
        queued = [a for a in assigns if not a["claimed"]]
        sidA = next(a["stripe_id"] for a in claimed if a["worker_id"] == A.worker_id)
        sidB = next(a["stripe_id"] for a in claimed if a["worker_id"] == B.worker_id)
        sidQ = queued[0]["stripe_id"]
        L0 = coord.ledger.get_stripe("run", sidA)["lease_expires_at"]

        def lease(sid):
            return coord.ledger.get_stripe("run", sid)["lease_expires_at"]

        # 1. WRONG WORKER — B claims progress on A's stripe.
        #    branch: accept_stripe_message "stale: stripe claimed_by ... != ..."
        assert coord._renew_active_lease(B, "run", sidA, B.worker_id,
                                         source="sub_stripe_result",
                                         now=2000.0) is False
        assert lease(sidA) == L0, "wrong worker renewed another worker's stripe"

        # 2. WRONG STRIPE — A sends progress naming B's stripe.
        #    branch: accept_stripe_message claimed_by mismatch
        assert coord._renew_active_lease(A, "run", sidB, A.worker_id,
                                         source="sub_stripe_result",
                                         now=2000.0) is False
        assert lease(sidB) == L0

        # 3. NOT COMPUTE-ACTIVE (queued backlog) — no lease exists to extend.
        #    branch: permitted-states (ST_CLAIMED,) vs a `pending` row
        assert coord._renew_active_lease(A, "run", sidQ, A.worker_id,
                                         source="sub_stripe_result",
                                         now=2000.0) is False
        assert lease(sidQ) is None, "a queued stripe acquired a lease"

        # 4. STALE ATTEMPT / late result from a prior attempt.
        #    branch: accept_stripe_message ledger attempt vs connection attempt
        A.assignment_attempts[sidA] = 0
        coord.ledger.set_stripe_fields("run", sidA, current_attempt=1)
        assert coord._renew_active_lease(A, "run", sidA, A.worker_id,
                                         source="sub_stripe_result",
                                         now=2000.0) is False
        assert lease(sidA) == L0, "a stale-attempt result renewed the lease"
        coord.ledger.set_stripe_fields("run", sidA, current_attempt=0)

        # 5. A STAGING stripe — compute is over; the slot and the lease are gone.
        coord.ledger.record_stripe_complete("run", sidB, 0, B.worker_id, 1, 0)
        assert coord.ledger.get_stripe("run", sidB)["state"] == ST_STAGING
        assert coord._renew_active_lease(B, "run", sidB, B.worker_id,
                                         source="sub_stripe_result",
                                         now=2000.0) is False
        assert lease(sidB) is None

        # 6. NO ACTIVE STRIPE NAMED (an idle heartbeat) — nothing to renew.
        assert coord._renew_active_lease(A, "run", None, A.worker_id,
                                         source="heartbeat", now=2000.0) is False
        assert coord._renew_active_lease(A, "run", "", A.worker_id,
                                         source="heartbeat", now=2000.0) is False

        # 7. `status` and `register` frames NEVER reach renewal at all — proven
        #    structurally by reading the live dispatch source, not by assertion:
        #    `_serve_dispatch` renews only in the `heartbeat` branch and in the
        #    `sub_stripe_result` branch after a successful insert.
        import inspect as _i
        src = _i.getsource(COORD.RangeMinerCoordinator._serve_dispatch)
        assert src.count("_renew_active_lease") == 2, (
            "exactly two renewal call sites are expected in _serve_dispatch "
            "(heartbeat + accepted sub_stripe_result); found "
            f"{src.count('_renew_active_lease')}")
        assert 'source="heartbeat"' in src and 'source="sub_stripe_result"' in src
        # and the ONE remaining live lease is still exactly where it started
        assert lease(sidA) == L0


# ===========================================================================
# G-F1-LIVE-STREAM-NO-EXPIRY   — THE RED-FIRST GATE (Beta §17)
# ===========================================================================
def g_live_stream_no_expiry():
    """THE 2026-08-09 GEOMETRY, REPRODUCED DETERMINISTICALLY.

    4 stripes per serial worker; delivery slower than the STAGE-WIDE lease; the
    worker never stops making progress. Two arms, same workload, same
    `compute_lease_timeout`, same stripe count, same worker count:

      PRE-FIX arm  — bulk claim at one `now`  -> the 4th stripe's lease expires
                     while the worker is mid-stream, and the constant-phase
                     matrix fails the trial. This is the incident.
      AMENDED arm  — active-lease scheduler   -> the SAME workload completes and
                     the matrix is never reached.

    Beta §17 forbids solving this by raising the timeout, cutting stripes, adding
    workers or weakening phase policy. None of those change between the arms —
    the assertions below pin every one of them."""
    STAGE_T, N = 1000.0, 4
    PER_STRIPE = 90.0          # 4 x 90 = 360 s of delivery > LEASE (300 s)
    assert N * PER_STRIPE > LEASE, "the geometry must exceed the stage-wide lease"

    # ---------------- PRE-FIX ARM (the mutant: bulk claim) ----------------
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp)
        A = _register(coord, "hostA:gpu0")
        coord.ledger.create_trial("run", 1, now=STAGE_T)
        for i in range(N):
            sid = f"run__st0_s{i}"
            coord.ledger.add_stripe("run", sid, i * MACRO, MACRO,
                                    "java_lcg", 1, now=STAGE_T)
        # bulk claim: every stripe stamped with ONE stage-wide deadline, which is
        # precisely what `assign_stripes` did before this amendment. Done through
        # set_stripe_fields so the F1 ledger guard does not stop us reproducing it.
        for i in range(N):
            coord.ledger.set_stripe_fields(
                "run", f"run__st0_s{i}", state=ST_CLAIMED,
                claimed_by=A.worker_id, current_attempt=0, expected_substripes=1,
                lease_expires_at=STAGE_T + LEASE)
            A.record_assignment(f"run__st0_s{i}", 0)
        # The worker works them serially and DELIVERS RESULTS THROUGHOUT — it is
        # visibly alive the whole time. Crucially there is NO renewal call here:
        # pre-amendment the only renewal path was the heartbeat, and the forensics
        # established that no heartbeat renewal ever landed on the three stripes
        # that expired (their `lease_expires_at` was still assign-time + 300 to the
        # microsecond). Modelling the pre-fix arm WITH renewal would model the fix.
        expired_mid_stream = None
        for i in range(N):
            t_start = STAGE_T + i * PER_STRIPE
            sid = f"run__st0_s{i}"
            for frac in (0.25, 0.5, 0.75):
                tick = t_start + PER_STRIPE * frac
                # real, accepted progress on the stripe being served right now
                coord.ledger.record_substripe_result(
                    "run", sid, 0, int(frac * 4), A.worker_id,
                    0, MACRO, 0, size_bytes=1, sha256="h" * 64)
                hits = [h["stripe_id"]
                        for h in coord.ledger.expired_claimed_stripes("run", tick)]
                if sid in hits and expired_mid_stream is None:
                    expired_mid_stream = (tick, sid, i)
                    break
            if expired_mid_stream is not None:
                # STOP HERE. The matrix must be run at the instant of detection,
                # while the stripe is still `claimed` and still streaming — running
                # it after the loop would find every stripe already `staging` and
                # return an empty list, which would look like a pass.
                break
            coord.ledger.record_stripe_complete("run", sid, 0, A.worker_id, 1, 0)
        assert expired_mid_stream is not None, (
            "PRE-FIX arm did not reproduce the defect — the geometry is wrong")
        tick, sid, idx = expired_mid_stream
        # the incident's exact shape: the stripe the worker is ACTIVELY streaming
        # has an expired lease, and it is one of the LATER stripes in its queue.
        assert idx == N - 1, f"expected the last queued stripe to expire, got {idx}"
        assert tick > STAGE_T + LEASE, (tick, STAGE_T + LEASE)
        assert coord.ledger.get_shards("run", sid, 0), \
            "the expired stripe had delivered no results — that is a dead worker, "\
            "not the defect"
        # and the constant-phase matrix would kill the trial on that input
        out = coord.process_lease_expiry("run", [A], now=tick)
        assert out and out[0]["action"] == "fail_trial", out
        assert coord.ledger.get_trial("run")["state"] == "aborted"
        assert coord.ledger.get_trial("run")["terminal_class"] == \
            TC_COMPUTE_LEASE_EXPIRY

    # ---------------- AMENDED ARM (identical workload) ----------------
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp)
        assert coord.config.compute_lease_timeout == LEASE, "timeout was raised"
        A = _register(coord, "hostA:gpu0")
        coord.ledger.create_trial("run", 1, now=STAGE_T)
        conns = [A]
        assigns = coord.assign_stripes("run", "java_lcg", 1, MACRO * N, conns,
                                       stripe_prefix="run__st0", now=STAGE_T)
        assert len(assigns) == N, "stripe count changed between the arms"
        assert len(conns) == 1, "worker count changed between the arms"
        t = STAGE_T
        for i in range(N):
            active = [s for s in coord.ledger.all_stripes("run")
                      if s["state"] == ST_CLAIMED]
            assert len(active) == 1, active
            sid = active[0]["stripe_id"]
            # deliver progress throughout the stripe's service, as the real
            # workers were doing when their leases expired
            for frac in (0.25, 0.5, 0.75, 1.0):
                tick = t + PER_STRIPE * frac
                coord._renew_active_lease(A, "run", sid, A.worker_id,
                                          source="sub_stripe_result", now=tick)
                # THE ASSERTION THIS GATE EXISTS FOR
                assert coord.ledger.expired_claimed_stripes("run", tick) == [], (
                    f"a continuously-progressing worker entered the lease-expiry "
                    f"matrix at t={tick} on {sid}")
                assert coord.process_lease_expiry("run", conns, now=tick) == []
            t += PER_STRIPE
            coord.ledger.record_stripe_complete("run", sid, 0, A.worker_id, 1, 0)
            coord.ledger.set_stripe_fields("run", sid, state=ST_DONE)
            coord.schedule_pending_stripes("run", "java_lcg", 1, conns,
                                           stage_prefix="run__st0", now=t)
        # the whole workload finished, past the stage-wide deadline, trial alive
        assert t == STAGE_T + N * PER_STRIPE > STAGE_T + LEASE
        assert coord.ledger.get_trial("run")["state"] == "running"
        done = [s for s in coord.ledger.all_stripes("run") if s["state"] == ST_DONE]
        assert len(done) == N, done


# ===========================================================================
# G-F1-DEAD-WORKER-STILL-EXPIRES  — the clean control
# ===========================================================================
def g_dead_worker_still_expires():
    """F1 did NOT disable fault detection. A genuinely silent active worker still
    expires, and a constant phase still fails the trial immediately."""
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp)
        A = _register(coord, "hostA:gpu0")
        coord.ledger.create_trial("run", 1, now=1000.0)
        conns, assigns = _stage(coord, n_workers=1, n_stripes=2, now=1000.0,
                                tmp_workers=[A])
        sid = next(a["stripe_id"] for a in assigns if a["claimed"])
        # NO heartbeat, NO progress — the worker is dead. Past the lease:
        t = 1000.0 + LEASE + 1.0
        assert [h["stripe_id"] for h in
                coord.ledger.expired_claimed_stripes("run", t)] == [sid]
        out = coord.process_lease_expiry("run", conns, now=t)
        assert out and out[0]["action"] == "fail_trial", out
        assert out[0]["reason"] == "constant_phase", out
        trial = coord.ledger.get_trial("run")
        assert trial["state"] == "aborted"
        assert trial["terminal_class"] == TC_COMPUTE_LEASE_EXPIRY, trial["terminal_class"]
        assert trial["terminal_stripe_id"] == sid
        assert trial["terminal_worker_id"] == A.worker_id


# ===========================================================================
# G-F1-HYBRID-MATRIX
# ===========================================================================
def g_hybrid_matrix():
    """A genuine hybrid active-stripe expiry still enters the certified retry /
    reassignment path — and the reassigned attempt gets a FRESH lease."""
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp)
        A = _register(coord, "hostA:gpu0")
        B = _register(coord, "hostB:gpu0")
        coord.ledger.create_trial("run", 1, now=1000.0)
        assigns = coord.assign_stripes("run", "java_lcg_hybrid", 3, MACRO * 2,
                                       [A, B], stripe_prefix="run__st0", now=1000.0)
        sid = next(a["stripe_id"] for a in assigns
                   if a["claimed"] and a["worker_id"] == A.worker_id)
        # A goes silent; B is busy with the other stripe.
        t = 1000.0 + LEASE + 1.0
        out = coord.process_lease_expiry("run", [A, B], now=t)
        assert out, out
        st = coord.ledger.get_stripe("run", sid)
        assert st["phase_degraded"] == 1, st
        assert st["current_attempt"] == 1, st
        assert coord.ledger.get_trial("run")["state"] == "running", \
            "a hybrid retryable expiry must NOT fail the trial"
        if out[0]["action"] == "requeued":
            # every alternate was busy — the stripe is queued, never lost
            assert st["state"] == ST_PENDING and st["claimed_by"] == A.worker_id
            coord.ledger.record_stripe_complete(
                "run", next(a["stripe_id"] for a in assigns
                            if a["claimed"] and a["worker_id"] == B.worker_id),
                0, B.worker_id, 1, 0)
            placed = coord.schedule_pending_stripes(
                "run", "java_lcg_hybrid", 3, [A, B],
                stage_prefix="run__st0", now=t + 10.0)
            assert placed and placed[0]["worker_id"] == B.worker_id, placed
            st = coord.ledger.get_stripe("run", sid)
        assert st["state"] == ST_CLAIMED
        assert st["claimed_by"] == B.worker_id, "reassigned to the SAME worker"
        assert st["lease_expires_at"] > t, "reassignment reused a stale deadline"
        # second failure on the new worker -> trial fails (matrix unchanged)
        t2 = st["lease_expires_at"] + 1.0
        out2 = coord.process_lease_expiry("run", [A, B], now=t2)
        assert any(o["action"] == "fail_trial" for o in out2), out2
        assert coord.ledger.get_trial("run")["state"] == "aborted"


# ===========================================================================
# G-F1-EXACT-STRIPE-COLLISION  (+ the prefix-as-exact mutant)
# ===========================================================================
def g_exact_stripe_collision():
    """[R1 BLOCKER A] The hybrid immediate-placement path selects by stripe
    IDENTITY, at the PRODUCTION 32-stripe geometry gate 12 will run.

    `G-F1-HYBRID-MATRIX` uses two stripes, so no `s1`/`s10` lexical sibling can
    exist and it cannot see this. Here the full stage exists: `run__st0_s1`'s
    retry is placed while `run__st0_s10 … s19` sit pending beside it. Under the
    pre-fix code the complete stripe id went into a LIKE-scoped prefix parameter,
    `s1%` also matched `s10 … s19`, and with every legitimate alternate busy and
    the prior claimer idle the result was: `s1` correctly skipped, an unrelated
    sibling claimed, and the non-empty result reported as
    `action="reassigned"` FOR `s1` — a stripe that was never reassigned."""
    N = 32
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp)
        FAM, PHASE = "java_lcg_hybrid", 3
        conns = [_register(coord, f"host{i}:gpu0") for i in range(3)]
        coord.ledger.create_trial("run", 1, now=1000.0)
        assigns = coord.assign_stripes("run", FAM, PHASE, MACRO * N, conns,
                                       stripe_prefix="run__st0", now=1000.0)
        assert len(assigns) == N, len(assigns)
        SID = "run__st0_s1"
        SIBLINGS = [f"run__st0_s1{d}" for d in range(10)]      # s10 … s19
        # the collision is REAL at this geometry: every sibling exists, and every
        # one of them is a pending row a prefix query would have swept in.
        ids = {s["stripe_id"] for s in coord.ledger.all_stripes("run")}
        assert SID in ids and set(SIBLINGS) <= ids, sorted(ids)
        assert all(coord.ledger.get_stripe("run", s)["state"] == ST_PENDING
                   for s in SIBLINGS)

        # ROLES, read from the ledger rather than assumed from round-robin order:
        # A = s1's claimer (it will be IDLE the instant its stripe is requeued);
        # the other two hold s0/s2 and are COMPUTE-BUSY throughout.
        A = next(c for c in conns
                 if c.worker_id == coord.ledger.get_stripe("run", SID)["claimed_by"])
        alternates = [c for c in conns if c is not A]
        assert len(alternates) == 2
        busy = coord.ledger.compute_busy_worker_ids("run")
        assert {w.worker_id for w in alternates} <= busy, busy

        def _sibling_states():
            return {s: (coord.ledger.get_stripe("run", s)["state"],
                        coord.ledger.get_stripe("run", s)["claimed_by"])
                    for s in SIBLINGS}
        before = _sibling_states()

        out = coord.handle_stripe_failure(
            "run", SID, retryable=True, eligible_workers=conns, now=1100.0)

        # 1. the failed stripe is queued, NOT reassigned
        assert out["action"] == "requeued", (
            f"expected 'requeued' (every alternate is compute-busy); got {out}")
        assert out["worker_id"] is None, out
        st = coord.ledger.get_stripe("run", SID)
        assert st["state"] == ST_PENDING, st
        assert st["current_attempt"] == 1, st
        assert st["claimed_by"] == A.worker_id, (
            "the prior claimer must survive on the pending row — it is what stops "
            "the scheduler handing the stripe straight back")
        # 2. NO sibling moved. This is the assertion the defect reds.
        assert _sibling_states() == before, (
            f"a lexical sibling changed state as a side effect of placing {SID}: "
            f"{ {k: v for k, v in _sibling_states().items() if before[k] != v} }")
        assert coord.ledger.get_trial("run")["state"] == "running"

        # 3. free a LEGITIMATE alternate -> s1 goes to it, with a FRESH lease,
        #    and never back to the prior claimer.
        other = alternates[0]
        osid = next(s["stripe_id"] for s in coord.ledger.all_stripes("run")
                    if s["claimed_by"] == other.worker_id
                    and s["state"] == ST_CLAIMED)
        coord.ledger.record_stripe_complete("run", osid, 0, other.worker_id, 1, 0)
        T2 = 1200.0
        placed = coord.schedule_pending_stripes(
            "run", FAM, PHASE, conns, stage_prefix="run__st0", now=T2)
        got = next((p for p in placed if p["stripe_id"] == SID), None)
        assert got is not None, f"{SID} was not placed on the freed alternate: {placed}"
        assert got["worker_id"] == other.worker_id, got
        st2 = coord.ledger.get_stripe("run", SID)
        assert st2["state"] == ST_CLAIMED
        assert st2["claimed_by"] == other.worker_id != A.worker_id, st2
        assert st2["lease_expires_at"] == T2 + LEASE, (
            f"reassignment did not stamp a fresh lease: {st2['lease_expires_at']} "
            f"!= {T2 + LEASE}")


def g_exact_stripe_collision_mutant():
    """MUTATION EVIDENCE for the exact selector. Restore ONLY prefix-as-exact —
    the pre-fix behaviour, expressed as routing `exact_stripe_id` back through
    the LIKE-scoped parameter — and prove (a) the mutated path EXECUTED and
    (b) it reds the credited assertions: a sibling is claimed and the retry is
    reported as `reassigned`."""
    N = 32
    executed = {"n": 0}
    orig = MinerLedger.pending_stripes

    def _mutant(self_, run_id, stage_prefix=None, *, exact_stripe_id=None):
        if exact_stripe_id is not None:
            executed["n"] += 1
            # THE DEFECT: identity handed to the prefix selector.
            return orig(self_, run_id, exact_stripe_id)
        return orig(self_, run_id, stage_prefix)

    def _drive():
        with tempfile.TemporaryDirectory() as tmp:
            coord = _coord(tmp)
            FAM, PHASE = "java_lcg_hybrid", 3
            conns = [_register(coord, f"host{i}:gpu0") for i in range(3)]
            coord.ledger.create_trial("run", 1, now=1000.0)
            coord.assign_stripes("run", FAM, PHASE, MACRO * N, conns,
                                 stripe_prefix="run__st0", now=1000.0)
            SID = "run__st0_s1"
            SIBLINGS = [f"run__st0_s1{d}" for d in range(10)]
            before = {s: coord.ledger.get_stripe("run", s)["state"]
                      for s in SIBLINGS}
            MinerLedger.pending_stripes = _mutant
            try:
                out = coord.handle_stripe_failure(
                    "run", SID, retryable=True, eligible_workers=conns, now=1100.0)
            finally:
                MinerLedger.pending_stripes = orig
            assert executed["n"] >= 1, (
                "the mutant was never called — the hybrid retry path does not go "
                "through the exact selector, so this gate proves nothing")
            after = {s: coord.ledger.get_stripe("run", s)["state"]
                     for s in SIBLINGS}
            assert after == before, (
                f"a lexical sibling changed state as a side effect: "
                f"{ {k: (before[k], after[k]) for k in after if after[k] != before[k]} }")
            assert out["action"] == "requeued", out

    _mutant_red(_drive, "prefix-as-exact selector restored")
    assert executed["n"] >= 1, "the mutant never executed"


# ===========================================================================
# G-F2-IDEMPOTENT-PARITY
# ===========================================================================
def g_f2_idempotent_parity():
    """[R1 BLOCKER B, Beta §7] THE FIRST DURABLE TERMINAL TRANSITION WINS
    TERMINAL IDENTITY PERMANENTLY.

    Abort with A, then abort the SAME run again with a deliberately
    contradictory B. The durable row, the ERROR log and every outward event must
    still be A; B must appear on NO surface. Exercised twice — the plain
    already-aborted path, and the RACE-SHAPED path where `mark_trial_aborted`
    returns False because another terminal transition won after this caller's
    initial read."""
    class _Sink:
        def __init__(self):
            self.aborts = []

        def publish_shard(self, m):
            pass

        def commit_trial(self, e):
            pass

        def abort_trial(self, e):
            self.aborts.append(e)

    A_REC = TerminalRecord(
        terminal_class=TC_COMPUTE_LEASE_EXPIRY,
        reason="AAA the first durable terminal transition",
        stripe_id="run__st0_s0", worker_id="hostA:gpu0", attempt=0)
    B_REC = TerminalRecord(
        terminal_class=TC_STRIPE_ERROR,
        reason="BBB a later contradictory proposal",
        stripe_id="run__st0_s9", worker_id="hostZ:gpu9", attempt=7)

    def _assert_is_A(ev, where):
        assert ev is not None, f"{where}: no event"
        assert ev["terminal_class"] == A_REC.terminal_class, (where, ev)
        assert ev["terminal_reason"] == A_REC.reason, (where, ev)
        assert ev["terminal_stripe_id"] == A_REC.stripe_id, (where, ev)
        assert ev["terminal_worker_id"] == A_REC.worker_id, (where, ev)
        assert ev["terminal_attempt"] == A_REC.attempt, (where, ev)
        blob = repr(ev)
        for tok in ("BBB", B_REC.terminal_class, B_REC.stripe_id,
                    B_REC.worker_id):
            assert tok not in blob, f"{where}: B reached an outward surface: {ev}"

    def _new(tmp):
        sink = _Sink()
        ledger = MinerLedger(os.path.join(tmp, "l.db"))
        coord = RangeMinerCoordinator(
            CoordinatorConfig(miner_stripe_size=MACRO,
                              compute_lease_timeout=LEASE,
                              staging_dir=os.path.join(tmp, "s")),
            ledger, phase5_sink=sink)
        ledger.create_trial("run", 1, now=1000.0)
        return coord, ledger, sink

    # ---- (i) the plain idempotent path ------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        coord, ledger, sink = _new(tmp)
        with _LogCapture() as cap:
            r1 = coord.abort_trial("run", reason="AAA", now=1100.0,
                                   terminal=A_REC)
        assert r1["first"] is True, r1
        _assert_is_A(r1["event"], "first abort return")
        # [R2] THE FIRST DELIVERY'S LEGACY `reason` IS ALSO CANONICAL.
        #
        # The caller's prose ("AAA") is DELIBERATELY different from the record's
        # ("AAA the first durable terminal transition"). That difference is the
        # whole point: this gate already constructed the counterexample and did
        # not assert it, so a first delivery carrying the caller's string and a
        # replay carrying the record's — same event_id, two payloads — went
        # unnoticed. Once a TerminalRecord exists it is the sole reason authority.
        assert r1["event"]["reason"] == A_REC.reason, (
            f"first delivery's legacy reason is not canonical: "
            f"{r1['event']['reason']!r} != {A_REC.reason!r}")
        assert len(sink.aborts) == 1, sink.aborts
        assert sink.aborts[0]["reason"] == A_REC.reason, (
            f"first SINK delivery's legacy reason is not canonical: "
            f"{sink.aborts[0]['reason']!r} != {A_REC.reason!r}")
        row = ledger.get_trial("run")
        assert row["terminal_class"] == A_REC.terminal_class
        assert row["terminal_reason"] == A_REC.reason
        first_logs = [m for m in cap.records if "TRIAL TERMINAL" in m]
        assert len(first_logs) == 1, first_logs
        assert "AAA" in first_logs[0] and "BBB" not in first_logs[0]

        with _LogCapture() as cap2:
            r2 = coord.abort_trial("run", reason="BBB", now=1200.0,
                                   terminal=B_REC)
        assert r2["first"] is False, r2
        # durable record UNCHANGED
        row2 = ledger.get_trial("run")
        assert row2["terminal_class"] == A_REC.terminal_class, row2
        assert row2["terminal_reason"] == A_REC.reason, row2
        assert row2["terminal_stripe_id"] == A_REC.stripe_id, row2
        assert row2["terminal_worker_id"] == A_REC.worker_id, row2
        assert row2["terminal_attempt"] == A_REC.attempt, row2
        # NO second terminal ERROR log
        assert [m for m in cap2.records if "TRIAL TERMINAL" in m] == [], (
            f"a second terminal ERROR record was emitted: {cap2.records}")
        # the returned event, and the replayed sink delivery, are BOTH A
        _assert_is_A(r2["event"], "idempotent abort return")
        assert r2["event"]["reason"] == A_REC.reason, (
            "the legacy prose field still carried the later caller's reason")
        assert len(sink.aborts) == 2, sink.aborts
        _assert_is_A(sink.aborts[1], "replayed sink delivery")
        # [R2] FULL PAYLOAD EQUALITY, not just event_id equality. This is the
        # assertion that states what an idempotent event identifier MEANS: the
        # same event identity carries the same event payload. event_id equality
        # alone was satisfied by two payloads that differed in `reason`.
        assert sink.aborts[1] == sink.aborts[0], (
            f"same event_id, different payload:\n"
            f"  first : {sink.aborts[0]}\n"
            f"  replay: {sink.aborts[1]}")
        assert sink.aborts[1]["event_id"] == sink.aborts[0]["event_id"]

    # ---- (ii) the RACE-SHAPED path ----------------------------------------
    # `mark_trial_aborted` returns False because another terminal transition won
    # AFTER this caller read state='running'.
    with tempfile.TemporaryDirectory() as tmp:
        coord, ledger, sink = _new(tmp)
        real_mark = ledger.mark_trial_aborted
        raced = {"n": 0}

        def _racing_mark(run_id, event_id, now=None, terminal=None):
            raced["n"] += 1
            # the competing transition lands FIRST, with A
            real_mark(run_id, "winner:abort", now, terminal=A_REC)
            return real_mark(run_id, event_id, now, terminal=terminal)

        ledger.mark_trial_aborted = _racing_mark
        with _LogCapture() as cap3:
            r3 = coord.abort_trial("run", reason="BBB", now=1300.0,
                                   terminal=B_REC)
        assert raced["n"] == 1, raced
        assert r3["first"] is False, (
            "the caller believed it won a race it lost")
        row3 = ledger.get_trial("run")
        assert row3["terminal_class"] == A_REC.terminal_class, row3
        assert row3["terminal_reason"] == A_REC.reason, row3
        assert row3["abort_event_id"] == "winner:abort", row3
        _assert_is_A(r3["event"], "race-lost abort return")
        # the losing caller adopts the WINNER's durable event identity too
        assert r3["event"]["event_id"] == "winner:abort", r3["event"]
        assert [m for m in cap3.records if "BBB" in m] == [], cap3.records
        assert len(sink.aborts) == 1, sink.aborts
        _assert_is_A(sink.aborts[0], "race-lost sink delivery")

    # ---- MUTANT: let the later caller keep terminal authority --------------
    def _m():
        with tempfile.TemporaryDirectory() as tmp:
            coord, ledger, sink = _new(tmp)
            coord.abort_trial("run", reason="AAA", now=1100.0, terminal=A_REC)
            orig_rebuild = RangeMinerCoordinator._terminal_from_trial_row
            RangeMinerCoordinator._terminal_from_trial_row = staticmethod(
                lambda row: B_REC)          # the pre-fix behaviour, exactly
            try:
                r = coord.abort_trial("run", reason="BBB", now=1200.0,
                                      terminal=B_REC)
            finally:
                RangeMinerCoordinator._terminal_from_trial_row = orig_rebuild
            _assert_is_A(r["event"], "mutant: idempotent abort return")
    _mutant_red(_m, "later caller keeps terminal authority")

    # ---- MUTANT [R2]: restore the FIRST-PATH REASON SPLIT ------------------
    # The pre-R2 shape exactly: the legacy `reason` key carries the CALLER's
    # prose on the first delivery while `terminal_reason` stays canonical. The
    # event dict handed to the sink IS the dict returned to the caller, so the
    # split lands on both surfaces just as it did in production.
    executed = {"n": 0}

    def _m_first_path_split():
        with tempfile.TemporaryDirectory() as tmp:
            coord, ledger, sink = _new(tmp)
            orig_abort = RangeMinerCoordinator.abort_trial

            def _split(self_, run_id, reason="", now=None, terminal=None):
                out = orig_abort(self_, run_id, reason, now, terminal)
                if out.get("first") and out.get("event") is not None:
                    executed["n"] += 1
                    out["event"]["reason"] = reason      # PRE-R2
                return out

            RangeMinerCoordinator.abort_trial = _split
            try:
                r_first = coord.abort_trial("run", reason="AAA", now=1100.0,
                                            terminal=A_REC)
            finally:
                RangeMinerCoordinator.abort_trial = orig_abort
            assert executed["n"] == 1, (
                "the mutant never ran — this gate would be vacuous")
            # each of the three R2 assertions must be able to catch it
            assert r_first["event"]["reason"] == A_REC.reason, (
                "R2-a: the first delivery's legacy reason is the caller's string")
            assert sink.aborts[0]["reason"] == A_REC.reason, (
                "R2-b: the first SINK delivery's legacy reason is the caller's")
            coord.abort_trial("run", reason="BBB", now=1200.0, terminal=B_REC)
            assert sink.aborts[1] == sink.aborts[0], (
                "R2-c: same event_id, different payload")
    _mutant_red(_m_first_path_split, "first-path reason split restored")
    assert executed["n"] == 1, "the R2 mutant never executed"


# ===========================================================================
# G-F1-BACKPRESSURE-HANDOFF
# ===========================================================================
def g_backpressure_handoff():
    """The certified S172-BP pause/resume lease protection stays green and now
    composes with progress renewal:
       coordinator-caused pause  -> exempt (never enters the matrix)
       resume + valid progress   -> the REAL renewal, and the grace ends
       resume + permanent silence-> still expires after the bounded grace"""
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp)
        A = _register(coord, "hostA:gpu0")
        coord.ledger.create_trial("run", 1, now=1000.0)
        conns, assigns = _stage(coord, n_workers=1, n_stripes=1, now=1000.0,
                                tmp_workers=[A])
        sid = assigns[0]["stripe_id"]
        t = 1000.0 + LEASE + 1.0

        # (a) PAUSED: coordinator-caused silence is exempt from the matrix.
        # Registered through the PRODUCTION entry point, so the gate exercises the
        # real registry rather than a hand-built dict that could drift from it.
        coord.register_paused_connection(object(), A.worker_id, now=1000.0)
        assert A.worker_id in coord.paused_worker_ids()
        assert coord.process_lease_expiry("run", conns, now=t) == [], \
            "a coordinator-paused worker entered the failure matrix"
        assert coord.ledger.get_trial("run")["state"] == "running"
        coord._paused_connections.clear()

        # (b) RESUMED with a grace bridge, then real progress -> grace cleared.
        # arm the grace bridge the way the resume path does (no public setter)
        with coord._pause_lock:
            coord._capacity_resume_grace[A.worker_id] = t + LEASE
        assert A.worker_id in coord.capacity_resume_grace(t)
        assert coord.process_lease_expiry("run", conns, now=t) == []
        assert coord._renew_active_lease(A, "run", sid, A.worker_id,
                                         source="sub_stripe_result",
                                         now=t) is True
        assert A.worker_id not in coord.capacity_resume_grace(t), \
            "the grace bridge outlived the real renewal it exists to cover"
        assert coord.ledger.get_stripe("run", sid)["lease_expires_at"] == t + LEASE

        # (c) permanently silent after resume -> still expires.
        t2 = t + LEASE + 1.0
        out = coord.process_lease_expiry("run", conns, now=t2)
        assert out and out[0]["action"] == "fail_trial", out
        assert coord.ledger.get_trial("run")["terminal_class"] == \
            TC_COMPUTE_LEASE_EXPIRY


# ===========================================================================
# G-F1-FROZEN-COHORT
# ===========================================================================
def g_frozen_cohort():
    """Dynamic one-at-a-time handoff does NOT reopen worker eligibility. A worker
    that registers after the cohort freeze may not receive pending work for that
    trial — on the initial handoff OR on any later scheduler pass.

    This is the positive behaviour the failed run demonstrated (22 late workers
    correctly excluded) and Beta §9 requires it to survive unchanged."""
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp)
        A = _register(coord, "hostA:gpu0")
        coord.ledger.create_trial("run", 1, now=1000.0)
        # freeze the cohort on A alone, exactly as preflight does
        coord.freeze_trial_cohort("run", {("java_lcg", 1): [A]})
        assigns = coord.assign_stripes("run", "java_lcg", 1, MACRO * 3, [A],
                                       stripe_prefix="run__st0", now=1000.0)
        assert sum(1 for a in assigns if a["claimed"]) == 1
        # a LATE worker joins and is offered to the scheduler
        LATE = _register(coord, "hostZ:gpu0", now=1100.0)
        placed = coord.schedule_pending_stripes(
            "run", "java_lcg", 1, [A, LATE], stage_prefix="run__st0", now=1100.0)
        assert placed == [], f"a post-freeze worker received pending work: {placed}"
        assert [s for s in coord.ledger.all_stripes("run")
                if s["claimed_by"] == "hostZ:gpu0"] == []
        # A frees its slot -> the backlog goes to A, still never to the late joiner
        sid = next(a["stripe_id"] for a in assigns if a["claimed"])
        coord.ledger.record_stripe_complete("run", sid, 0, A.worker_id, 1, 0)
        placed2 = coord.schedule_pending_stripes(
            "run", "java_lcg", 1, [A, LATE], stage_prefix="run__st0", now=1200.0)
        assert len(placed2) == 1 and placed2[0]["worker_id"] == A.worker_id, placed2


# ===========================================================================
# G-F1-ABORT-PENDING
# ===========================================================================
def g_abort_pending():
    """Abort a trial holding BOTH claimed work and pending backlog; prove no
    nonterminal/runnable stripe remains, and that a LATER scheduler pass cannot
    claim one of the cancelled rows."""
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp)
        A = _register(coord, "hostA:gpu0")
        coord.ledger.create_trial("run", 1, now=1000.0)
        conns, assigns = _stage(coord, n_workers=1, n_stripes=6, now=1000.0,
                                tmp_workers=[A])
        assert sum(1 for a in assigns if a["claimed"]) == 1
        assert sum(1 for a in assigns if not a["claimed"]) == 5
        coord.fail_trial("run", reason="test abort",
                         terminal=TerminalRecord(
                             terminal_class=TC_STRIPE_ERROR, reason="test abort"))
        states = {s["state"] for s in coord.ledger.all_stripes("run")}
        assert states == {ST_CANCELLED}, states
        for st in coord.ledger.all_stripes("run"):
            assert st["state"] not in (ST_PENDING, ST_CLAIMED, ST_STAGING)
        # a later scheduler pass must not resurrect anything
        placed = coord.schedule_pending_stripes(
            "run", "java_lcg", 1, conns, stage_prefix="run__st0", now=2000.0)
        assert placed == [], f"scheduler claimed a row after termination: {placed}"
        # and the ledger primitive refuses even a direct claim
        first = coord.ledger.all_stripes("run")[0]["stripe_id"]
        coord.ledger.set_stripe_fields("run", first, state=ST_PENDING,
                                       claimed_by=None)
        assert coord.ledger.claim_stripe("run", first, A.worker_id, 0, 1,
                                         3000.0) is False, \
            "claim_stripe honoured a claim on a terminated trial"
        assert coord.ledger.get_stripe("run", first)["state"] == ST_PENDING


# ===========================================================================
# G-F2-TERMINAL-DURABILITY  (+ two mutants)
# ===========================================================================
def g_f2_terminal_durability():
    """Inject a genuine lease failure; the SAME canonical class/reason/stripe/
    worker/attempt must appear in (1) durable trial state, (2) the abort event,
    (3) the coordinator ERROR log. A mutation dropping the durable reason or the
    log must red."""
    class _Sink:
        def __init__(self):
            self.aborts = []

        def publish_shard(self, m):
            pass

        def commit_trial(self, e):
            pass

        def abort_trial(self, e):
            self.aborts.append(e)

    def _run(mutate_ledger=False, mutate_log=False):
        with tempfile.TemporaryDirectory() as tmp:
            sink = _Sink()
            ledger = MinerLedger(os.path.join(tmp, "l.db"))
            coord = RangeMinerCoordinator(
                CoordinatorConfig(miner_stripe_size=MACRO,
                                  compute_lease_timeout=LEASE,
                                  staging_dir=os.path.join(tmp, "s")),
                ledger, phase5_sink=sink)
            if mutate_ledger:
                orig = ledger.mark_trial_aborted
                ledger.mark_trial_aborted = (
                    lambda r, e, n=None, terminal=None: orig(r, e, n, None))
            A = _register(coord, "hostA:gpu0")
            coord.ledger.create_trial("run", 1, now=1000.0)
            assigns = coord.assign_stripes("run", "java_lcg", 1, MACRO, [A],
                                           stripe_prefix="run__st0", now=1000.0)
            sid = assigns[0]["stripe_id"]
            t = 1000.0 + LEASE + 1.0
            with _LogCapture() as cap:
                if mutate_log:
                    _saved = COORD.logger.error
                    COORD.logger.error = lambda *a, **k: None
                    try:
                        coord.process_lease_expiry("run", [A], now=t)
                    finally:
                        COORD.logger.error = _saved
                else:
                    coord.process_lease_expiry("run", [A], now=t)
            trial = ledger.get_trial("run")
            return trial, sink.aborts, cap.records, sid, A.worker_id

    # ---- the real, unmutated run: all three surfaces agree ----
    trial, aborts, logs, sid, wid = _run()
    assert trial["state"] == "aborted"
    # (1) durable
    assert trial["terminal_class"] == TC_COMPUTE_LEASE_EXPIRY
    assert trial["terminal_stripe_id"] == sid
    assert trial["terminal_worker_id"] == wid
    assert trial["terminal_attempt"] == 0
    assert trial["terminal_reason"] and "compute lease" in trial["terminal_reason"]
    # (2) the abort event
    assert len(aborts) == 1, aborts
    ev = aborts[0]
    assert ev["terminal_class"] == trial["terminal_class"]
    assert ev["terminal_reason"] == trial["terminal_reason"]
    assert ev["terminal_stripe_id"] == trial["terminal_stripe_id"]
    assert ev["terminal_worker_id"] == trial["terminal_worker_id"]
    assert ev["terminal_attempt"] == trial["terminal_attempt"]
    # (3) the ERROR log — same class, stripe, worker, and the same reason text
    hits = [m for m in logs if "TRIAL TERMINAL" in m]
    assert len(hits) == 1, f"expected exactly one terminal ERROR record, got {hits}"
    line = hits[0]
    assert f"class={TC_COMPUTE_LEASE_EXPIRY}" in line, line
    assert sid in line and wid in line, line
    assert trial["terminal_reason"] in line, "the log re-composed its own reason"
    # ATOMICITY: aborted with a reason, never aborted-with-NULL on a path that had one
    assert not (trial["state"] == "aborted" and trial["terminal_class"] is None)

    # ---- MUTANT 1: drop the durable record ----
    def _m1():
        tr, _ev, _lg, _s, _w = _run(mutate_ledger=True)
        assert tr["terminal_class"] is not None, "durable terminal record dropped"
    _mutant_red(_m1, "durable terminal record dropped")

    # ---- MUTANT 2: drop the ERROR log ----
    def _m2():
        _tr, _ev, lg, _s, _w = _run(mutate_log=True)
        assert [m for m in lg if "TRIAL TERMINAL" in m], "terminal ERROR log dropped"
    _mutant_red(_m2, "terminal ERROR log dropped")


# ===========================================================================
def main():
    print("=" * 70)
    print("S172 F1/F2 — ACTIVE-LEASE SCHEDULER + TERMINAL OBSERVABILITY")
    print("=" * 70)
    _check("G-F1-QUEUE-NO-LEASE       backlog has no claimer and no lease",
           g_queue_no_lease)
    _check("G-F1-ONE-ACTIVE           one compute claim/worker (+bulk mutant)",
           g_one_active)
    _check("G-F1-FRESH-HANDOFF        lease stamped at handoff, not planning",
           g_fresh_handoff)
    _check("G-F1-PROGRESS-RENEWAL     accepted result extends the active lease",
           g_progress_renewal)
    _check("G-F1-HEARTBEAT-RENEWAL    heartbeat renewal survives, real dispatch",
           g_heartbeat_renewal)
    _check("G-F1-SCOPE-RENEWAL        every forbidden renewal is refused",
           g_scope_renewal)
    _check("G-F1-LIVE-STREAM-NO-EXPIRY  RED-FIRST: the 2026-08-09 geometry",
           g_live_stream_no_expiry)
    _check("G-F1-DEAD-WORKER-STILL-EXPIRES  clean control: detection intact",
           g_dead_worker_still_expires)
    _check("G-F1-HYBRID-MATRIX        hybrid expiry still retries once",
           g_hybrid_matrix)
    _check("G-F1-EXACT-STRIPE-COLLISION  32-stripe geometry: s1 vs s10-s19",
           g_exact_stripe_collision)
    _check("G-F1-EXACT-STRIPE-COLLISION/M  prefix-as-exact mutant must red",
           g_exact_stripe_collision_mutant)
    _check("G-F2-IDEMPOTENT-PARITY    first durable transition owns identity",
           g_f2_idempotent_parity)
    _check("G-F1-BACKPRESSURE-HANDOFF pause/grace composes with renewal",
           g_backpressure_handoff)
    _check("G-F1-FROZEN-COHORT        late worker gets no pending work",
           g_frozen_cohort)
    _check("G-F1-ABORT-PENDING        no runnable row survives termination",
           g_abort_pending)
    _check("G-F2-TERMINAL-DURABILITY  one record, three surfaces (+2 mutants)",
           g_f2_terminal_durability)

    ok = sum(1 for _n, p, _t in _results if p)
    print("=" * 70)
    print(f"{ok}/{len(_results)} checks green")
    if ok != len(_results):
        print("\nFAILURES (DO NOT COMMIT):\n")
        for n, p, tb in _results:
            if not p:
                print(f"--- {n} ---\n{tb}")
        return 1
    print("All checks green — S172 F1/F2 active-lease scheduler + terminal "
          "observability (pending Team Beta review).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
