#!/usr/bin/env python3
"""
test_s172_elapsed_roundtrip.py — G-ELAPSED-ROUNDTRIP

Implements the single gate required by PART 2 of
`docs/CLAUDE_CODE_INSTRUCTIONS_STAGING_CAPACITY_AMENDMENT.md`, which carries Team
Beta's ruling *"STEP-1 SEARCH GEOMETRY…"* (2026-08-08) **R4**.

WHY THIS FILE IS SEPARATE FROM test_s172_staging_backpressure.py
---------------------------------------------------------------
The brief requires Part 2 to stay liftable: *"Write it so it can be lifted out and
submitted independently if Part 1 needs another review round. Its gate must not
depend on Part 1's changes."* A gate living in the Part 1 suite would inherit that
suite's fixtures and its capacity semantics, and its independence would then be a
property of MY INSPECTION rather than of the file. Here independence is structural:
this module imports nothing from the Part 1 suite, constructs its own ledger, and
touches no staging-capacity surface. Delete Part 1 entirely and this still runs.

THE DEFECT
----------
The worker computes and transmits `StripeCompleteMessage.elapsed_s`
(`miner/range_miner_worker.py:1345` -> `miner/range_miner_protocol.py`), and the
coordinator dropped it: the dispatch call site passed only `substripes_done` and
`survivors_total`, and `record_stripe_complete` had no elapsed parameter. A
schema-wide search for `elapsed|duration|compute|started_at` returned no column.

⚠ MEASUREMENT CAVEAT (Beta R4, binding on every consumer of this column)
`elapsed_s` is stripe SERVICE TIME — sufficient for per-stripe and per-worker rate
calculations and for sizing work. It is NOT aggregate cluster wall-clock
throughput: concurrent worker intervals OVERLAP. Do not reconstruct fleet
throughput by summing or averaging per-stripe seeds/sec; any fleet-level figure
needs an overlap-aware makespan denominator.

Run:
    cd ~/distributed_prng_analysis
    source ~/venvs/torch/bin/activate
    PYTHONPATH=. python3 -u tests/test_s172_elapsed_roundtrip.py \
        | tee /tmp/s172_elapsed_roundtrip.log
"""
import os
import sqlite3
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
    except Exception as e:                                    # noqa: BLE001
        _results.append((name, False, traceback.format_exc()))
        print(f"  [{_FAIL}] {name}: {e}")


from miner.range_miner_coordinator import (  # noqa: E402
    ST_CLAIMED,
    ST_STAGING,
    CoordinatorConfig,
    MinerLedger,
    NodeConfig,
    RangeMinerCoordinator,
)
from miner.range_miner_protocol import (  # noqa: E402
    StripeCompleteMessage,
    from_dict,
    message_from_bytes,
    message_to_bytes,
)

CAPS = {"amd": 2_000_000, "nvidia": 5_000_000,
        "amd_hybrid": 1_000_000, "nvidia_hybrid": 2_500_000}
VARIANTS = ["java_lcg", "java_lcg_reverse",
            "java_lcg_hybrid", "java_lcg_hybrid_reverse"]
SPOOL_ROOT = "/var/spool/miner"

WID = "hostA:gpu0"
RUN = "runE"
SID = "runE_s0"

# The worker's own measurement. Deliberately NOT a plausible test-runtime value:
# this gate completes in milliseconds, so a coordinator-side clock could never
# produce 12345.678. Storing it verbatim is therefore behavioural proof that the
# persisted number came off the WIRE and was not synthesized locally (scope item 3).
WORKER_ELAPSED = 12345.678


# ===========================================================================
# Harness — a real coordinator over a real ledger; no Part 1 surfaces touched
# ===========================================================================
def _coord(tmp, dbname="elapsed.db"):
    ledger = MinerLedger(os.path.join(tmp, dbname))
    return RangeMinerCoordinator(
        CoordinatorConfig(staging_dir=os.path.join(tmp, "staging")), ledger)


def _register(coord, wid=WID, backend="cuda", now=100.0):
    node = NodeConfig(hostname=wid.split(":")[0], spool_root=SPOOL_ROOT,
                      ssh_address="10.0.0.9", ssh_user="michael")
    return coord.register_worker(
        worker_id=wid, hostname=wid.split(":")[0], backend=backend,
        capabilities={"seed_caps": dict(CAPS),
                      "supported_variants": list(VARIANTS)},
        node_config=node, now=now)


def _claim(coord, conn, run_id=RUN, sid=SID, wid=WID, expected=2, now=100.0):
    """Claim one stripe. `expected`=2 with zero shards keeps the L8 predicate at
    'incomplete, still waiting' (substripes_match False), so finalize_stripe parks
    the stripe in staging and the retry matrix is never reached — this gate is
    about persistence, not completion."""
    coord.ledger.add_stripe(run_id, sid, 0, 30, "java_lcg", 1, now)
    coord.ledger.claim_stripe(run_id, sid, wid, 0, expected, 1e9)
    conn.record_assignment(sid, 0)


def _deliver(coord, conn, msg, run_id=RUN):
    """worker -> WIRE -> coordinator. The message is serialized to real framed
    bytes and decoded back, so the gate exercises the actual transport contract
    (length-prefixed compact JSON + the from_dict field filter), not an in-process
    object handoff that would prove nothing about the wire."""
    wire = message_to_bytes(msg)
    decoded, _ = message_from_bytes(wire)
    coord._serve_dispatch(decoded, run_id, msg.worker_id, {msg.worker_id: conn},
                          lambda: [conn])
    return decoded


def _stored(coord, run_id=RUN, sid=SID):
    return coord.ledger.get_stripe(run_id, sid)["elapsed_s"]


# ===========================================================================
# G-ELAPSED-ROUNDTRIP
# ===========================================================================
def arm_roundtrip_reaches_ledger_unmodified():
    """worker -> wire -> coordinator -> ledger, value byte-for-byte intact."""
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp)
        conn = _register(coord)
        _claim(coord, conn)
        _deliver(coord, conn, StripeCompleteMessage(
            worker_id=WID, stripe_id=SID, substripes_done=2,
            survivors_total=7, elapsed_s=WORKER_ELAPSED))
        got = _stored(coord)
        assert got == WORKER_ELAPSED, (
            f"worker reported {WORKER_ELAPSED} but ledger holds {got!r}")
        assert isinstance(got, float), f"stored type is {type(got).__name__}, not float"
        # The transition itself must still be the one the pre-amendment code made.
        assert coord.ledger.get_stripe(RUN, SID)["state"] == ST_STAGING


def arm_absent_field_stores_null_not_zero():
    """A pre-R4 peer omits the key entirely -> SQL NULL, never 0.0."""
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp)
        conn = _register(coord)
        _claim(coord, conn)
        # Built as a raw dict with NO elapsed_s key — exactly what an older peer
        # puts on the wire. from_dict() applies the dataclass default.
        raw = {"message_type": "stripe_complete", "worker_id": WID,
               "stripe_id": SID, "substripes_done": 2, "survivors_total": 7}
        assert "elapsed_s" not in raw
        decoded = from_dict(raw)
        assert decoded.elapsed_s is None, (
            f"absent field decoded as {decoded.elapsed_s!r}; it must be None so "
            f"'not reported' stays distinguishable from a real 0.0")
        coord._serve_dispatch(decoded, RUN, WID, {WID: conn}, lambda: [conn])
        got = _stored(coord)
        assert got is None, f"absent elapsed_s stored as {got!r}, expected NULL"


def arm_genuine_zero_is_not_null():
    """The control that makes the NULL arm meaningful (VIR-2).

    A gate that only proves 'absent -> NULL' also passes an implementation that
    stores NULL unconditionally. A worker that genuinely measured 0.0 must persist
    0.0, and 0.0 must NOT collapse to NULL."""
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp)
        conn = _register(coord)
        _claim(coord, conn)
        _deliver(coord, conn, StripeCompleteMessage(
            worker_id=WID, stripe_id=SID, substripes_done=2,
            survivors_total=7, elapsed_s=0.0))
        got = _stored(coord)
        assert got is not None, "a genuine 0.0 measurement collapsed to NULL"
        assert got == 0.0, f"genuine zero stored as {got!r}"


def arm_replay_is_idempotent_and_cannot_corrupt():
    """A replayed completion must neither double-write nor overwrite the value.

    Idempotency here is INHERITED, not added: the UPDATE is guarded on
    state=ST_CLAIMED, which the first call clears to ST_STAGING. The replay is
    given a DIFFERENT elapsed_s so the assertion distinguishes 'the replay was
    refused' from 'the replay happened to rewrite the same number'."""
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp)
        conn = _register(coord)
        _claim(coord, conn)
        first = StripeCompleteMessage(
            worker_id=WID, stripe_id=SID, substripes_done=2,
            survivors_total=7, elapsed_s=WORKER_ELAPSED)
        _deliver(coord, conn, first)
        assert _stored(coord) == WORKER_ELAPSED

        # Exact duplicate.
        _deliver(coord, conn, first)
        assert _stored(coord) == WORKER_ELAPSED, "exact replay mutated the value"

        # Hostile replay: same identity, different measurement.
        _deliver(coord, conn, StripeCompleteMessage(
            worker_id=WID, stripe_id=SID, substripes_done=2,
            survivors_total=7, elapsed_s=999.999))
        got = _stored(coord)
        assert got == WORKER_ELAPSED, (
            f"a replayed completion overwrote the first measurement: {got!r}")

        # And the ledger primitive itself reports the refusal, not a silent no-op.
        again = coord.ledger.record_stripe_complete(
            RUN, SID, 0, WID, 2, 7, elapsed_s=555.5)
        assert again is False, "replayed record_stripe_complete claimed a transition"
        assert _stored(coord) == WORKER_ELAPSED


def arm_additive_migration_leaves_old_rows_null():
    """Scope item 1: ONE ADDITIVE column; old rows may remain NULL.

    A ledger created before the column existed must gain it on open — `CREATE
    TABLE IF NOT EXISTS` is a no-op on an existing DB, so without the migration
    every write would raise — and its pre-existing rows must read NULL."""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "pre_r4.db")
        # A pre-R4 stripes table: the committed schema MINUS elapsed_s.
        con = sqlite3.connect(db)
        con.execute("""
            CREATE TABLE stripes (
                run_id TEXT NOT NULL, stripe_id TEXT NOT NULL,
                seed_start INTEGER NOT NULL, seed_count INTEGER NOT NULL,
                state TEXT NOT NULL DEFAULT 'pending', claimed_by TEXT,
                current_attempt INTEGER NOT NULL DEFAULT 0,
                staging_generation INTEGER NOT NULL DEFAULT 0,
                expected_substripes INTEGER, lease_expires_at REAL,
                phase INTEGER NOT NULL DEFAULT 0,
                family_name TEXT NOT NULL DEFAULT '',
                phase_degraded INTEGER NOT NULL DEFAULT 0,
                stripe_complete_seen INTEGER NOT NULL DEFAULT 0,
                substripes_done INTEGER, survivors_total INTEGER,
                created_at REAL NOT NULL,
                PRIMARY KEY (run_id, stripe_id))""")
        con.execute("INSERT INTO stripes (run_id, stripe_id, seed_start, "
                    "seed_count, created_at) VALUES ('old','old_s0',0,10,1.0)")
        con.commit()
        cols_before = {r[1] for r in con.execute("PRAGMA table_info(stripes)")}
        con.close()
        assert "elapsed_s" not in cols_before, "fixture is not a pre-R4 schema"

        ledger = MinerLedger(db)                      # _init_db runs the migration
        row = ledger.get_stripe("old", "old_s0")
        assert row is not None, "the pre-existing row did not survive the migration"
        assert "elapsed_s" in row, "the additive column was not added on open"
        assert row["elapsed_s"] is None, (
            f"backfilled old row to {row['elapsed_s']!r}; it must remain NULL")

        # Re-opening must not attempt the ALTER a second time.
        ledger2 = MinerLedger(db)
        assert ledger2.get_stripe("old", "old_s0")["elapsed_s"] is None


def arm_coordinator_does_not_synthesize():
    """Scope item 3: persist the WORKER-reported value; never substitute a
    coordinator-side timestamp.

    Behavioural, not textual: the ledger is driven with a value no local clock
    could produce, and separately with None. If the coordinator ever supplied its
    own measurement, the None case would come back as a number."""
    with tempfile.TemporaryDirectory() as tmp:
        coord = _coord(tmp)
        conn = _register(coord)
        _claim(coord, conn)
        coord.ledger.record_stripe_complete(RUN, SID, 0, WID, 2, 7, elapsed_s=None)
        got = _stored(coord)
        assert got is None, (
            f"coordinator synthesized {got!r} for a completion that reported "
            f"nothing — a coordinator clock is a DIFFERENT measurement")


def main():
    print("=" * 74)
    print("G-ELAPSED-ROUNDTRIP — S172 elapsed_s persistence (Beta R4)")
    print("=" * 74)
    _check("G-ELAPSED-ROUNDTRIP: worker->wire->coordinator->ledger, unmodified",
           arm_roundtrip_reaches_ledger_unmodified)
    _check("G-ELAPSED-ROUNDTRIP: absent field stores NULL, not 0.0",
           arm_absent_field_stores_null_not_zero)
    _check("G-ELAPSED-ROUNDTRIP: a genuine 0.0 is stored, not collapsed to NULL",
           arm_genuine_zero_is_not_null)
    _check("G-ELAPSED-ROUNDTRIP: replay is idempotent and cannot overwrite",
           arm_replay_is_idempotent_and_cannot_corrupt)
    _check("G-ELAPSED-ROUNDTRIP: additive migration leaves old rows NULL",
           arm_additive_migration_leaves_old_rows_null)
    _check("G-ELAPSED-ROUNDTRIP: coordinator never synthesizes a replacement",
           arm_coordinator_does_not_synthesize)

    print()
    print("=" * 74)
    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    for name, ok, tb in _results:
        if not ok:
            print(f"\n--- {name} ---\n{tb}")
    print(f"\n{passed}/{total} checks green")
    if passed == total:
        print("COMPLETION SENTINEL: PASS — G-ELAPSED-ROUNDTRIP green")
        return 0
    print("COMPLETION SENTINEL: FAIL — G-ELAPSED-ROUNDTRIP red")
    return 1


if __name__ == "__main__":
    sys.exit(main())
