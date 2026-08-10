#!/usr/bin/env python3
"""
GATE SUITE — gate-12 concurrency sampler measures the POST-F1 state model
=========================================================================
NOT a certified S172 suite. New file, no fleet, no coordinator, no port bind.
Every gate runs against a SYNTHETIC ledger built here.

WHAT THIS PROVES
  G1  N claimed / M pending / K staging -> reports N compute-active, M queued
  G2  staging is NOT counted as occupancy
  G3  ESTAB is NOT counted as occupancy
  G4  25 claimed but ZERO pending          -> NOT SATISFYING
  G5  25 distinct workers only ACROSS instants -> NOT SATISFYING
  G6  25 simultaneous WITH queue remaining -> SATISFYING   (clean control)
  G7  queries are scoped to the run under observation
  G8  the run is latched only if created after sampling started
  G9  the connection is genuinely read-only
  G10 a production database is refused outright
  M1  MUTATION: the 2026-08-09 query -- state IN ('claimed','staging'), no
      pending term -- must red G1

REVISION 1 ARMS (Beta "PRE-RERUN ITEMS REVIEW" §3, §4, §5)
  A1  ATOMICITY, clean control: a writer commits BETWEEN the sample's two reads
      and the emitted sample is still self-consistent -- one instant
  A2  ATOMICITY, mutation: with `read_snapshot` neutered to a no-op (two
      autocommit reads, the pre-R1 shape) the SAME interleaving produces a
      sample whose occupancy and counts describe different instants
  A3  the snapshot is RELEASED: the next sample on the same connection sees the
      committed change -- a held read transaction would freeze the sampler
  B1  ss missing        -> UNAVAILABLE, and specifically not 0
  B2  ss non-zero exit  -> UNAVAILABLE   (this is the path that returned 0)
  B3  ss unparseable    -> UNAVAILABLE
  B4  ss timeout        -> UNAVAILABLE
  B5  ss observed zero  -> 0, NOT laundered into UNAVAILABLE (converse control)
  B6  the TSV renders UNAVAILABLE, never 0 or -1
  B7  the verdict is byte-identical whether ESTAB is 0, 400 or UNAVAILABLE
  C1  25 claimed / 7 pending held STATIC -> simultaneity SATISFIED,
      turnover NOT satisfied
  C2  pending drains under full occupancy -> BOTH satisfied
  C3  25 only across different instants   -> NEITHER
  C4  pending flat but stripes reach done/staging -> turnover satisfied on the
      transition term alone
  C5  the two verdicts are rendered separately and are never collapsed

DEFECT-B ARMS — ALL-WINDOW TURNOVER AGGREGATION (Beta "GATE-12 ATTEMPT-2
FORENSIC RULING" §§17-19). Turnover was aggregated over the SINGLE LONGEST
qualifying window; the Gate-12 question is existential over ALL of them.

Beta numbers these gates B1-B5. They are named DB1-DB5 in this file because the
ESTAB arms above already own the labels B1-B7, and two arms answering to "B1" in
one evidence log is a defect in the log. DBn IS Beta's Bn, one for one.
  DB1 longest qualifying window static, a SHORTER one holds turnover -> BOTH
      satisfied.  THE DEFECT REPRODUCTION: reds under the longest-only
      aggregation, and the in-arm mutant restoring longest-only reds with it
  DB2 pending drains while occupancy DIPS below threshold -> not credited, and
      still not credited once the aggregation quantifies over both windows
  DB3 an UNOBSERVED gap separates the movement -> neither window may borrow the
      other's; a window broken by a gap is two windows
  DB4 several qualifying windows, all static -> NOT satisfied.  THE CONTROL:
      the fix must not turn a genuinely static run into a false positive
  DB5 several windows hold turnover -> the witness is the documented
      deterministic choice (earliest, NOT longest), labelled, and stable

RUN: source ~/venvs/torch/bin/activate && python3 -u tests/test_gate12_concurrency_sampler.py
"""

import contextlib
import json
import os
import shutil
import sqlite3
import stat
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import gate12_concurrency_sampler as S  # noqa: E402

GREEN, RED, YELLOW, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"
_RESULTS = []


def check(name, ok, detail=""):
    tag = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    _RESULTS.append((name, bool(ok)))
    print(f"  [{tag}] {name:<32} {detail}")


# ──────────────────────────────────────────────────────────────────────────────
# synthetic ledger
# ──────────────────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE stripes (
    run_id TEXT NOT NULL, stripe_id TEXT NOT NULL,
    seed_start INTEGER NOT NULL, seed_count INTEGER NOT NULL,
    state TEXT NOT NULL DEFAULT 'pending', claimed_by TEXT,
    created_at REAL NOT NULL,
    PRIMARY KEY (run_id, stripe_id));
"""


def build_ledger(spec, created_at=1000.0, wal=False):
    """spec: {run_id: [(state, claimed_by, count), ...]}

    `wal=True` matches the production ledger's journal mode, which is what lets
    a writer commit while a reader holds an open read transaction. The atomicity
    arms need it: in rollback-journal mode the writer would simply block on the
    reader's shared lock and the interleaving under test could not occur at all.
    """
    path = Path(tempfile.mkdtemp(prefix="tfm_sampler_")) / "miner_ledger.db"
    conn = sqlite3.connect(path)
    if wal:
        conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    i = 0
    for run_id, rows in spec.items():
        for state, worker, count in rows:
            for _ in range(count):
                conn.execute(
                    "INSERT INTO stripes VALUES (?,?,?,?,?,?,?)",
                    (run_id, f"{run_id}__s{i}", i * 100, 100, state, worker,
                     created_at))
                i += 1
    conn.commit()
    conn.close()
    return str(path)


def samples_from(seq, base=1000.0, interval=2.0):
    """seq: [(compute_active_worker_ids, queued_pending[, done[, staging]]), ...]
    -> sample dicts. The two trailing terms default to 0, so every pre-R1 call
    site keeps its exact meaning."""
    out = []
    for k, item in enumerate(seq):
        workers, queued = item[0], item[1]
        done = item[2] if len(item) > 2 else 0
        staging = item[3] if len(item) > 3 else 0
        ws = set(workers)
        out.append({
            "epoch": base + k * interval,
            "ts_iso": f"T+{k * interval:.0f}s",
            "compute_active": len(ws), "active_workers": ws,
            "queued_pending": queued, "claimed_rows": len(ws),
            "staging": staging, "done": done, "cancelled": 0, "failed": 0,
        })
    return out


def verdict_line(text, label):
    """The rendered value of one labelled verdict, read out of the summary the
    way an operator reads it."""
    for line in text.splitlines():
        if line.startswith(label):
            return line.split(":", 1)[1].strip()
    return None


# ──────────────────────────────────────────────────────────────────────────────
# G1-G3: the state model
# ──────────────────────────────────────────────────────────────────────────────

def g1_g2_g3_state_model():
    """N=12 claimed (12 distinct workers), M=20 pending, K=7 staging held by
    SEVEN FURTHER distinct workers. The old query would answer 19."""
    claimed = [(S.ST_CLAIMED, f"rig:gpu{i}", 1) for i in range(12)]
    staging = [(S.ST_STAGING, f"rig:gpu{100 + i}", 1) for i in range(7)]
    spec = {"runA": claimed + staging + [(S.ST_PENDING, None, 20),
                                         (S.ST_DONE, "rig:gpu0", 3)]}
    path = build_ledger(spec)
    with S.connect_ro(path) as conn:
        s = S.sample_run(conn, "runA")

    check("G1-COMPUTE-ACTIVE-AND-QUEUE",
          s["compute_active"] == 12 and s["queued_pending"] == 20,
          f"compute_active={s['compute_active']} queued={s['queued_pending']}")

    check("G2-STAGING-NOT-OCCUPANCY",
          s["compute_active"] == 12 and s["staging"] == 7
          and not (s["active_workers"] & {f"rig:gpu{100 + i}" for i in range(7)}),
          f"staging={s['staging']} reported separately; occupancy still 12")

    # ESTAB never reaches the verdict: evaluate() has no such input, so a huge
    # connection count cannot manufacture a satisfying window.
    v = S.evaluate(samples_from([({f"w{i}" for i in range(3)}, 20)] * 5),
                   threshold=25, min_window_samples=2)
    check("G3-ESTAB-NOT-OCCUPANCY",
          v["satisfied"] is False and "estab" not in v,
          "3 busy workers stay 3 regardless of connection count")


# ──────────────────────────────────────────────────────────────────────────────
# G4-G6: the verdict
# ──────────────────────────────────────────────────────────────────────────────

def g4_full_but_empty_queue():
    """25 simultaneously compute-active, ZERO queued. Beta: insufficient."""
    w25 = {f"w{i}" for i in range(25)}
    v = S.evaluate(samples_from([(w25, 0)] * 10), 25, 2)
    text = S.render_summary(v, "runA", "t0", "t1", "/x/miner_ledger.db")
    check("G4-NO-QUEUE-NOT-SATISFIED",
          v["satisfied"] is False and v["peak_simultaneous_active"] == 25
          and v["samples_satisfying"] == 0 and "NOT SATISFIED" in text
          and "queue was empty" in text,
          f"peak={v['peak_simultaneous_active']} satisfying={v['samples_satisfying']}")


def g5_union_across_instants():
    """25 distinct workers, but never more than 5 at once. The union is 25 and
    must NOT satisfy — this is the max-over-time error, made explicit."""
    seq = [({f"w{5 * k + j}" for j in range(5)}, 12) for k in range(5)]
    v = S.evaluate(samples_from(seq), 25, 2)
    text = S.render_summary(v, "runA", "t0", "t1", "/x/miner_ledger.db")
    check("G5-UNION-NOT-SIMULTANEITY",
          v["satisfied"] is False
          and v["distinct_workers_union"] == 25
          and v["peak_simultaneous_active"] == 5
          and "NOT SATISFIED" in text
          and "CANNOT satisfy the criterion" in text,
          f"union={v['distinct_workers_union']} peak_simultaneous="
          f"{v['peak_simultaneous_active']}")


def g6_clean_control():
    """VIR-2 clean control: the criterion genuinely met must report SATISFIED,
    or every negative above is vacuous.

    Under R1 this fixture drains 7 -> 5 -> 3 while occupancy holds at 25, so it
    is also the clean control for verdict 2 (C2)."""
    w25 = {f"w{i}" for i in range(25)}
    seq = [({"w0"}, 30), (w25, 7), (w25, 5), (w25, 3), ({"w0", "w1"}, 0)]
    v = S.evaluate(samples_from(seq), 25, 2)
    text = S.render_summary(v, "runA", "t0", "t1", "/x/miner_ledger.db")
    check("G6-CLEAN-CONTROL-SATISFIED",
          v["satisfied"] is True
          and v["longest_window_samples"] == 3
          and v["longest_window_min_active"] == 25
          and v["longest_window_min_queued"] == 3
          and v["queued_at_peak"] in (7, 5, 3)
          and verdict_line(text, S.LABEL_SIMULTANEITY) == "SATISFIED",
          f"window={v['longest_window_samples']} samples "
          f"{v['longest_window_seconds']:.0f}s min_queued="
          f"{v['longest_window_min_queued']}")


def g6b_single_sample_blip():
    """One satisfying instant is a blip, not a window, at the default setting."""
    w25 = {f"w{i}" for i in range(25)}
    v = S.evaluate(samples_from([({"w0"}, 9), (w25, 4), ({"w0"}, 9)]), 25, 2)
    check("G6B-BLIP-IS-NOT-A-WINDOW",
          v["satisfied"] is False and v["longest_window_samples"] == 1
          and v["samples_satisfying"] == 1,
          f"longest_window={v['longest_window_samples']} sample")


# ──────────────────────────────────────────────────────────────────────────────
# G7-G10: scoping and safety
# ──────────────────────────────────────────────────────────────────────────────

def g7_run_scoping():
    """A second run in the same ledger must not leak into the observation."""
    spec = {
        "runOBSERVED": [(S.ST_CLAIMED, f"a:gpu{i}", 1) for i in range(4)]
                       + [(S.ST_PENDING, None, 6)],
        "runOTHER": [(S.ST_CLAIMED, f"b:gpu{i}", 1) for i in range(30)]
                    + [(S.ST_PENDING, None, 99)],
    }
    path = build_ledger(spec)
    with S.connect_ro(path) as conn:
        s = S.sample_run(conn, "runOBSERVED")
    check("G7-SCOPED-TO-RUN",
          s["compute_active"] == 4 and s["queued_pending"] == 6,
          f"neighbouring run of 30/99 ignored; got {s['compute_active']}/"
          f"{s['queued_pending']}")


def g8_latch_after_start():
    """Rows predating the sampler belong to an earlier run and must not latch."""
    path = build_ledger({"runOLD": [(S.ST_PENDING, None, 4)]}, created_at=500.0)
    with S.connect_ro(path) as conn:
        before = S.discover_run_id(conn, since_epoch=1000.0)
    conn = sqlite3.connect(path)
    conn.execute("INSERT INTO stripes VALUES ('runNEW','runNEW__s0',0,10,"
                 "'pending',NULL,1500.0)")
    conn.commit(); conn.close()
    with S.connect_ro(path) as conn:
        after = S.discover_run_id(conn, since_epoch=1000.0)
    check("G8-LATCH-AFTER-START",
          before is None and after == "runNEW",
          f"pre-start={before!r} post-start={after!r}")


def g9_connection_is_readonly():
    path = build_ledger({"runA": [(S.ST_PENDING, None, 1)]})
    try:
        with S.connect_ro(path) as conn:
            conn.execute("UPDATE stripes SET state='claimed'")
            conn.commit()
        check("G9-READ-ONLY", False, "a write SUCCEEDED through the sampler conn")
    except sqlite3.OperationalError as e:
        check("G9-READ-ONLY", "readonly" in str(e).lower(), str(e))


def g10_production_db_refused():
    tmp = Path(tempfile.mkdtemp(prefix="tfm_sampler_"))
    prod = tmp / "prng_analysis.db"
    sqlite3.connect(prod).close()
    try:
        S.guard_db_path(str(prod))
        check("G10-PROD-DB-REFUSED", False, "prng_analysis.db was ACCEPTED")
    except SystemExit as e:
        check("G10-PROD-DB-REFUSED", "production database" in str(e), str(e)[:70])
    # and a non-existent ledger is never created
    try:
        S.guard_db_path(str(tmp / "nope.db"))
        check("G10B-NO-DB-CREATION", False, "missing ledger accepted")
    except SystemExit as e:
        check("G10B-NO-DB-CREATION", "never creates" in str(e), str(e)[:70])


# ──────────────────────────────────────────────────────────────────────────────
# A: ONE SAMPLE IS ONE SNAPSHOT  (Beta R1 §3)
# ──────────────────────────────────────────────────────────────────────────────

class _InterleavingConn:
    """A real read-only connection that fires ONE writer commit in the gap
    between the sample's two reads.

    That gap is the defect's whole habitat: read 1 is the occupancy set, read 2
    is the queue depth, and under two autocommit reads a stripe transitioning in
    between yields a sample whose halves describe different instants. The
    interleaving is triggered off the second statement's own SQL, so if the
    production query is ever reshaped the trigger stops matching -- `fired` is
    asserted by every arm precisely so that turns into a red, not a vacuous
    pass (VIR-2).
    """

    _READ2_MARKER = "GROUP BY state"

    def __init__(self, conn, mutate):
        self._conn = conn
        self._mutate = mutate
        self.fired = False

    def execute(self, sql, *a, **k):
        if not self.fired and self._READ2_MARKER in sql:
            self._mutate()
            self.fired = True
        return self._conn.execute(sql, *a, **k)

    def __getattr__(self, name):
        return getattr(self._conn, name)


def _writer(path, sql):
    def go():
        w = sqlite3.connect(path, timeout=5.0)
        try:
            w.execute(sql)
            w.commit()
        finally:
            w.close()
    return go


def _atomicity_fixture():
    """25 compute-active workers (one claim each, as F1 enforces) and 7 queued.
    The interleaved write completes three of them."""
    spec = {"runATOM": [(S.ST_CLAIMED, f"w{i}", 1) for i in range(25)]
                       + [(S.ST_PENDING, None, 7)]}
    path = build_ledger(spec, wal=True)
    mutate = _writer(path, "UPDATE stripes SET state='done' "
                           "WHERE run_id='runATOM' AND claimed_by IN "
                           "('w0','w1','w2')")
    return path, mutate


def _self_consistent(s):
    """Under F1 there is exactly ONE compute-active claim per serial worker, so
    the size of the occupancy set and the count of `claimed` rows are the same
    quantity measured by the two different reads. A sample that straddles a
    transition disagrees with itself; a snapshot cannot."""
    return s["compute_active"] == s["claimed_rows"]


def a1_a3_snapshot_is_atomic():
    path, mutate = _atomicity_fixture()
    with S.connect_ro(path) as conn:
        w = _InterleavingConn(conn, mutate)
        s = S.sample_run(w, "runATOM")
        # and, after the snapshot is released, the very next sample must see it
        after = S.sample_run(conn, "runATOM")

    total = s["claimed_rows"] + s["queued_pending"] + s["done"]
    check("A1-SNAPSHOT-IS-ONE-INSTANT",
          w.fired
          and _self_consistent(s)
          and s["compute_active"] == 25 and s["claimed_rows"] == 25
          and s["done"] == 0 and s["queued_pending"] == 7 and total == 32,
          f"interleaved_write={w.fired} active={s['compute_active']} "
          f"claimed={s['claimed_rows']} done={s['done']} "
          f"pending={s['queued_pending']}")

    check("A3-SNAPSHOT-RELEASED",
          after["compute_active"] == 22 and after["claimed_rows"] == 22
          and after["done"] == 3 and _self_consistent(after),
          f"next sample sees the commit: active={after['compute_active']} "
          f"done={after['done']}")


def a2_mutation_two_autocommit_reads():
    """MUTATION: neuter `read_snapshot` to a no-op and the production code is
    back to two independent autocommit read transactions -- the pre-R1 shape,
    verbatim, since nothing else about `sample_run` changes. The SAME
    interleaving must now produce a self-inconsistent sample."""
    path, mutate = _atomicity_fixture()
    saved = S.read_snapshot
    S.read_snapshot = contextlib.nullcontext          # two autocommit reads
    try:
        with S.connect_ro(path) as conn:
            w = _InterleavingConn(conn, mutate)
            s = S.sample_run(w, "runATOM")
    finally:
        S.read_snapshot = saved

    check("A2-MUTANT-STRADDLES-INSTANTS",
          w.fired
          and not _self_consistent(s)
          and s["compute_active"] == 25      # read 1: before the commit
          and s["claimed_rows"] == 22        # read 2: after it
          and s["done"] == 3,
          f"occupancy={s['compute_active']} from one instant vs "
          f"claimed_rows={s['claimed_rows']} done={s['done']} from another")


# ──────────────────────────────────────────────────────────────────────────────
# MUTATION — the 2026-08-09 query
# ──────────────────────────────────────────────────────────────────────────────

def m1_legacy_query_reds_g1():
    """The pre-fix sampler asked
         count(distinct claimed_by) WHERE state IN ('claimed','staging')
       and never selected `pending`. On the SAME fixture it answers 19, not 12,
       and cannot produce a queue depth at all."""
    claimed = [(S.ST_CLAIMED, f"rig:gpu{i}", 1) for i in range(12)]
    staging = [(S.ST_STAGING, f"rig:gpu{100 + i}", 1) for i in range(7)]
    path = build_ledger({"runA": claimed + staging + [(S.ST_PENDING, None, 20)]})

    with S.connect_ro(path) as conn:
        legacy = conn.execute(
            "select count(distinct claimed_by) from stripes "
            "where state in ('claimed','staging') and claimed_by is not null"
        ).fetchone()[0]
        fixed = S.sample_run(conn, "runA")

    check("M1A-LEGACY-OVERSTATES", legacy == 19 and fixed["compute_active"] == 12,
          f"legacy={legacy} (overstates by 7 staging) vs fixed="
          f"{fixed['compute_active']}")
    check("M1B-LEGACY-BLIND-TO-QUEUE",
          fixed["queued_pending"] == 20,
          "legacy query selected no pending term at all; fixed reports 20")


# ──────────────────────────────────────────────────────────────────────────────
# B: THE ESTAB COUNT IS HONEST OR ABSENT  (Beta R1 §4)
# ──────────────────────────────────────────────────────────────────────────────

SS_EXIT1 = "#!/usr/bin/env bash\necho 'ss: bad filter' >&2\nexit 1\n"
SS_GARBLED = "#!/usr/bin/env bash\necho 'wat'\n"
SS_HANG = "#!/usr/bin/env bash\nsleep 30\n"
SS_EMPTY = "#!/usr/bin/env bash\nexit 0\n"
SS_THREE = ("#!/usr/bin/env bash\n"
            "echo '0      0      192.168.3.177:5700   192.168.3.122:41001'\n"
            "echo '0      0      192.168.3.177:5700   192.168.3.156:41002'\n"
            "echo '0      0      192.168.3.177:5700   192.168.3.164:41003'\n")


@contextlib.contextmanager
def fake_ss(body):
    """A controlled PATH holding one `ss`, or none at all when body is None.

    Behavioural, not stubbed at the Python seam: `estab_observation` runs
    unmodified and unaware, and the real subprocess call, exit status, stdout
    and parse are all genuinely exercised.

    `body is None` REPLACES PATH with an empty directory — a real /usr/bin/ss
    further down would rescue the missing-binary arm and make it vacuous. Every
    other arm PREPENDS, so the shim shadows the real `ss` while still being able
    to run (its own interpreter and `sleep` are found the normal way).
    """
    tmp = Path(tempfile.mkdtemp(prefix="tfm_ss_"))
    binroot = tmp / "bin"
    binroot.mkdir()
    saved = os.environ.get("PATH", "")
    if body is None:
        os.environ["PATH"] = str(binroot)
    else:
        p = binroot / "ss"
        p.write_text(body)
        p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        os.environ["PATH"] = f"{binroot}:{saved}"
    try:
        yield
    finally:
        os.environ["PATH"] = saved
        shutil.rmtree(tmp, ignore_errors=True)


def b1_b5_estab_outcomes():
    with fake_ss(None):
        missing = S.estab_observation(5700)
    check("B1-SS-MISSING-UNAVAIL",
          missing["estab"] is None and missing["estab"] != 0
          and missing["estab_status"] == S.ESTAB_UNAVAILABLE
          and missing["estab_reason"] == "ss_not_found",
          f"estab={missing['estab']!r} reason={missing['estab_reason']}")

    with fake_ss(SS_EXIT1):
        failed = S.estab_observation(5700)
    check("B2-SS-NONZERO-EXIT-UNAVAIL",
          failed["estab"] is None
          and failed["estab_status"] == S.ESTAB_UNAVAILABLE
          and failed["estab_reason"].startswith("ss_exit_1"),
          f"estab={failed['estab']!r} reason={failed['estab_reason']}")

    with fake_ss(SS_GARBLED):
        garbled = S.estab_observation(5700)
    check("B3-SS-UNPARSEABLE-UNAVAIL",
          garbled["estab"] is None
          and garbled["estab_status"] == S.ESTAB_UNAVAILABLE
          and garbled["estab_reason"].startswith("unparseable_ss_output"),
          f"estab={garbled['estab']!r} reason={garbled['estab_reason']}")

    saved = S.ESTAB_TIMEOUT_SECONDS
    S.ESTAB_TIMEOUT_SECONDS = 1
    try:
        with fake_ss(SS_HANG):
            timed_out = S.estab_observation(5700)
    finally:
        S.ESTAB_TIMEOUT_SECONDS = saved
    check("B4-SS-TIMEOUT-UNAVAIL",
          timed_out["estab"] is None
          and timed_out["estab_status"] == S.ESTAB_UNAVAILABLE
          and timed_out["estab_reason"] == "timeout",
          f"estab={timed_out['estab']!r} reason={timed_out['estab_reason']}")

    # Converse control: an observed zero must NOT be laundered into UNAVAILABLE,
    # or the fix would merely move the dishonesty to the other end.
    with fake_ss(SS_EMPTY):
        zero = S.estab_observation(5700)
    with fake_ss(SS_THREE):
        three = S.estab_observation(5700)
    check("B5-OBSERVED-ZERO-IS-ZERO",
          zero["estab"] == 0 and zero["estab_status"] == S.ESTAB_OK
          and zero["estab_reason"] is None
          and three["estab"] == 3 and three["estab_status"] == S.ESTAB_OK,
          f"nothing-established={zero['estab']} three-established={three['estab']}")


def b6_tsv_never_renders_zero():
    """The evidence FILE is where the dishonesty would land. An unavailable
    ESTAB must reach the TSV as UNAVAILABLE -- not 0, and not the old -1
    sentinel, which is still a number an analyst would read as a count."""
    base = samples_from([({f"w{i}" for i in range(25)}, 7)])[0]
    unavail = dict(base, estab=None, estab_status=S.ESTAB_UNAVAILABLE,
                   estab_reason="ss_not_found")
    observed = dict(base, estab=0, estab_status=S.ESTAB_OK, estab_reason=None)

    row_u = S.format_tsv_row(unavail, "runA", True).rstrip("\n").split("\t")
    row_o = S.format_tsv_row(observed, "runA", True).rstrip("\n").split("\t")
    i_estab = S.TSV_COLUMNS.index("estab")
    i_reason = S.TSV_COLUMNS.index("estab_reason")

    check("B6-TSV-RENDERS-UNAVAILABLE",
          len(row_u) == len(S.TSV_COLUMNS)
          and row_u[i_estab] == S.ESTAB_UNAVAILABLE
          and row_u[i_estab] not in ("0", "-1", "None")
          and row_u[i_reason] == "ss_not_found"
          and row_o[i_estab] == "0" and row_o[i_reason] == "-",
          f"unavailable -> {row_u[i_estab]!r}/{row_u[i_reason]!r}; "
          f"observed zero -> {row_o[i_estab]!r}")


def b7_verdict_unaffected_by_estab():
    """ESTAB is not a criterion term. Three runs identical but for ESTAB --
    absent, zero, and an absurd 400 -- must produce the SAME verdict object and
    the same two rendered verdicts."""
    w25 = {f"w{i}" for i in range(25)}
    seq = [({"w0"}, 9), (w25, 7), (w25, 4), (w25, 2)]

    def run(estab_value, status, reason):
        rows = samples_from(seq)
        for r in rows:
            r.update(estab=estab_value, estab_status=status, estab_reason=reason)
        v = S.evaluate(rows, 25, 2)
        text = S.render_summary(v, "runA", "t0", "t1", "/x/db",
                                estab=S.summarize_estab(rows))
        return v, text

    v_unavail, t_unavail = run(None, S.ESTAB_UNAVAILABLE, "ss_not_found")
    v_zero, t_zero = run(0, S.ESTAB_OK, None)
    v_many, t_many = run(400, S.ESTAB_OK, None)

    same_verdict = v_unavail == v_zero == v_many
    lines = [(verdict_line(t, S.LABEL_SIMULTANEITY),
              verdict_line(t, S.LABEL_TURNOVER))
             for t in (t_unavail, t_zero, t_many)]
    check("B7-VERDICT-UNAFFECTED-BY-ESTAB",
          same_verdict and len(set(lines)) == 1
          and lines[0] == ("SATISFIED", "SATISFIED")
          and S.ESTAB_UNAVAILABLE in t_unavail
          and "ss_not_found" in t_unavail
          and "max=400" in t_many,
          f"verdicts identical across estab in (None, 0, 400): {lines[0]}")


# ──────────────────────────────────────────────────────────────────────────────
# C: TURNOVER IS A SECOND, SEPARATE VERDICT  (Beta R1 §5)
# ──────────────────────────────────────────────────────────────────────────────

def c1_static_queue_no_turnover():
    """THE arm this item exists for. 25 claimed and 7 pending, frozen: the fleet
    is full and the queue is non-empty for the whole window, so simultaneity is
    genuinely satisfied -- and nothing was ever consumed, so turnover is not."""
    w25 = {f"w{i}" for i in range(25)}
    seq = [({"w0"}, 7)] + [(w25, 7)] * 5 + [({"w0"}, 7)]
    v = S.evaluate(samples_from(seq), 25, 2)
    text = S.render_summary(v, "runA", "t0", "t1", "/x/db")
    check("C1-STATIC-QUEUE-NO-TURNOVER",
          v["satisfied"] is True
          and v["turnover_satisfied"] is False
          and v["turnover_pending_delta"] == 0
          and v["turnover_transitions"] == 0
          and verdict_line(text, S.LABEL_SIMULTANEITY) == "SATISFIED"
          and verdict_line(text, S.LABEL_TURNOVER) == "NOT SATISFIED"
          and "WHY NOT (verdict 2)" in text,
          f"window={v['longest_window_samples']} samples, pending delta="
          f"{v['turnover_pending_delta']}, transitions="
          f"{v['turnover_transitions']}")


def c2_draining_queue_both():
    """Clean control for verdict 2: pending drains 7 -> 1 while occupancy holds
    at 25 and stripes reach done. Both criteria satisfied."""
    w25 = {f"w{i}" for i in range(25)}
    seq = [({"w0"}, 7, 0), (w25, 7, 0), (w25, 5, 2), (w25, 3, 4), (w25, 1, 6)]
    v = S.evaluate(samples_from(seq), 25, 2)
    text = S.render_summary(v, "runA", "t0", "t1", "/x/db")
    check("C2-DRAINING-QUEUE-BOTH",
          v["satisfied"] is True
          and v["turnover_satisfied"] is True
          and v["turnover_pending_delta"] == 6
          and v["turnover_done_delta"] == 6
          and v["turnover_transitions"] == 6
          and verdict_line(text, S.LABEL_SIMULTANEITY) == "SATISFIED"
          and verdict_line(text, S.LABEL_TURNOVER) == "SATISFIED"
          and "WHY NOT" not in text,
          f"pending delta={v['turnover_pending_delta']} "
          f"transitions={v['turnover_transitions']}")


def c3_union_only_neither():
    """The retained existing arm, now scored against both criteria: 25 distinct
    workers across instants, never 25 at once. No window, so nothing to measure
    turnover in -- NEITHER is satisfied, and turnover says why."""
    seq = [({f"w{5 * k + j}" for j in range(5)}, 12) for k in range(5)]
    v = S.evaluate(samples_from(seq), 25, 2)
    text = S.render_summary(v, "runA", "t0", "t1", "/x/db")
    check("C3-UNION-ONLY-NEITHER",
          v["satisfied"] is False
          and v["turnover_satisfied"] is False
          and v["turnover_pending_delta"] is None
          and "no qualifying window" in v["turnover_reason"]
          and verdict_line(text, S.LABEL_SIMULTANEITY) == "NOT SATISFIED"
          and verdict_line(text, S.LABEL_TURNOVER) == "NOT SATISFIED",
          f"union={v['distinct_workers_union']} peak="
          f"{v['peak_simultaneous_active']} turnover={v['turnover_reason'][:34]}")


def c4_transitions_alone_suffice():
    """The 'and/or' matters. Across a stage boundary the backlog is replenished,
    so pending can stay flat while stripes are genuinely being consumed. The
    transition term must carry the verdict on its own."""
    w25 = {f"w{i}" for i in range(25)}
    seq = [(w25, 7, 0, 0), (w25, 7, 2, 1), (w25, 7, 5, 2)]
    v = S.evaluate(samples_from(seq), 25, 2)
    check("C4-TRANSITIONS-ALONE-SUFFICE",
          v["satisfied"] is True
          and v["turnover_satisfied"] is True
          and v["turnover_pending_delta"] == 0      # NOT the drain term
          and v["turnover_done_delta"] == 5
          and v["turnover_transitions"] == 7,       # done+staging step-deltas
          f"pending flat ({v['turnover_pending_delta']}) but "
          f"{v['turnover_transitions']} transitions, {v['turnover_done_delta']} done")


def c5_verdicts_are_not_collapsed():
    """Beta: 'Do not collapse the two into a single pass/fail.' Both labels must
    appear, exactly once each, on separate lines -- and the exit status must
    still tell the two failure modes apart."""
    w25 = {f"w{i}" for i in range(25)}
    static = S.evaluate(samples_from([({"w0"}, 7)] + [(w25, 7)] * 4), 25, 2)
    drain = S.evaluate(samples_from([({"w0"}, 7), (w25, 7), (w25, 4), (w25, 2)]),
                       25, 2)
    none_ = S.evaluate(samples_from([({"w0"}, 7)] * 4), 25, 2)
    t = S.render_summary(static, "runA", "t0", "t1", "/x/db")

    def rc(v):
        """The exit-status mapping from main(), stated once here so a collapse
        back to one boolean reds."""
        if not v["satisfied"]:
            return 2
        return 0 if v["turnover_satisfied"] else 3

    check("C5-VERDICTS-NOT-COLLAPSED",
          t.count(S.LABEL_SIMULTANEITY) == 1
          and t.count(S.LABEL_TURNOVER) == 1
          and verdict_line(t, S.LABEL_SIMULTANEITY) != verdict_line(t, S.LABEL_TURNOVER)
          and (rc(drain), rc(static), rc(none_)) == (0, 3, 2),
          f"both/turnover-only/neither -> exit {rc(drain)}/{rc(static)}/{rc(none_)}")


# ──────────────────────────────────────────────────────────────────────────────
# D: A FAILED LEDGER READ IS UNOBSERVED, NEVER ZERO OCCUPANCY  (Beta R2 §1)
# ──────────────────────────────────────────────────────────────────────────────

def gap(sample, reason="OperationalError:database is locked"):
    """Turn a built sample into an UNOBSERVED one, the way the loop does."""
    return S.unobserved_row(sample["epoch"], sample["ts_iso"], reason)


@contextlib.contextmanager
def failing_connect(fail_on):
    """Make `connect_ro` raise on the given 1-based call indices.

    Call 1 is `assert_is_ledger`'s startup probe; the sampling loop starts at
    call 2. Failing mid-run is the point — a ledger that was readable at arm
    time and then locks is exactly the production case."""
    real = S.connect_ro
    state = {"n": 0}

    def go(path):
        state["n"] += 1
        if state["n"] in fail_on:
            raise sqlite3.OperationalError("database is locked")
        return real(path)

    S.connect_ro = go
    try:
        yield state
    finally:
        S.connect_ro = real


def run_main(ledger, out, summary, extra=()):
    """Drive the real main() and capture its stdout."""
    import io
    argv = ["--ledger", ledger, "--out", out, "--summary", summary,
            "--threshold", "25", "--min-window-samples", "2",
            "--interval", "0.05", "--max-seconds", "0.5",
            "--quiesce-seconds", "999", "--port", "5700", *extra]
    buf, old = io.StringIO(), sys.stdout
    sys.stdout = buf
    try:
        rc = S.main(argv)
    finally:
        sys.stdout = old
    return rc, buf.getvalue()


def _static_saturated_ledger():
    """25 compute-active / 7 queued, held static — satisfying, no turnover."""
    spec = {"runGAP": [(S.ST_CLAIMED, f"w{i}", 1) for i in range(25)]
                      + [(S.ST_PENDING, None, 7)]}
    return build_ledger(spec, created_at=1.0, wal=True)


def d1_injected_read_failure_is_unobserved():
    """END TO END through main(): a ledger read fails mid-run and the emitted
    evidence must say so, not report an idle fleet."""
    path = _static_saturated_ledger()
    tmp = Path(path).parent
    out, summ = str(tmp / "c.tsv"), str(tmp / "v.txt")
    with failing_connect({3, 4}):
        rc, _log = run_main(path, out, summ, extra=["--run-id", "runGAP"])

    rows = [l.rstrip("\n").split("\t")
            for l in open(out).read().splitlines()[1:]]
    cols = {c: i for i, c in enumerate(S.TSV_COLUMNS)}
    unobs = [r for r in rows if r[cols["obs_status"]] == S.OBS_UNOBSERVED]
    obs = [r for r in rows if r[cols["obs_status"]] == S.OBS_OBSERVED]
    text = open(summ).read()

    ledger_cols = [cols[f] for f in S.LEDGER_FIELDS]
    renders_unobserved = all(r[c] == S.OBS_UNOBSERVED
                             for r in unobs for c in ledger_cols)
    never_zero = all(r[c] != "0" for r in unobs for c in ledger_cols)
    satisfies_blank = all(r[cols["satisfies"]] == "-" for r in unobs)

    check("D1-READ-FAILURE-IS-UNOBSERVED",
          len(unobs) == 2 and len(obs) >= 2
          and renders_unobserved and never_zero and satisfies_blank
          and f"UNOBSERVED      : {len(unobs)}" in text
          and f"samples emitted   : {len(rows)}" in text
          and f"OBSERVED        : {len(obs)}" in text
          and "OperationalError" in text,
          f"{len(unobs)} unobserved / {len(obs)} observed rows; ledger columns "
          f"render {S.OBS_UNOBSERVED}, satisfies='-', census in summary (rc={rc})")


def d2_gap_breaks_the_window():
    """THE gap rule. Five satisfying samples with ONE unobserved sample in the
    middle: as a single run they would clear a 4-sample minimum; broken at the
    gap they are runs of 2 and 3 and neither does. The control is the identical
    fixture with the gap observed."""
    w25 = {f"w{i}" for i in range(25)}
    built = samples_from([(w25, 7)] * 6)
    with_gap = list(built)
    with_gap[2] = gap(built[2])

    v_gap = S.evaluate(with_gap, 25, 4)
    v_cont = S.evaluate(built, 25, 4)

    check("D2-GAP-BREAKS-THE-WINDOW",
          v_gap["satisfied"] is False
          and v_gap["window_count"] == 2
          and v_gap["longest_window_samples"] == 3
          and v_gap["samples_unobserved"] == 1
          and v_cont["satisfied"] is True          # clean control
          and v_cont["longest_window_samples"] == 6,
          f"with gap: {v_gap['window_count']} runs, longest "
          f"{v_gap['longest_window_samples']} < 4 -> NOT SATISFIED; "
          f"same fixture without the gap -> SATISFIED")


def d3_unobserved_never_enters_as_zero():
    """The census must have a KNOWN denominator, and an unobserved sample must
    not be counted as an observation of an empty fleet."""
    w25 = {f"w{i}" for i in range(25)}
    built = samples_from([(w25, 7)] * 4)
    mixed = [built[0], built[1], gap(built[2]), gap(built[3])]
    v = S.evaluate(mixed, 25, 2)
    text = S.render_summary(v, "runA", "t0", "t1", "/x/db", interval=2.0)

    check("D3-UNOBSERVED-NOT-A-ZERO-SAMPLE",
          v["samples_total"] == 4
          and v["samples_observed"] == 2
          and v["samples_unobserved"] == 2
          and v["samples_satisfying"] == 2          # not 2/4
          and v["peak_simultaneous_active"] == 25   # gaps excluded from peak
          and v["distinct_workers_union"] == 25
          and "2/2 observed" in text
          and v["unobserved_reasons"] == ["OperationalError"],
          f"total={v['samples_total']} observed={v['samples_observed']} "
          f"unobserved={v['samples_unobserved']} satisfying="
          f"{v['samples_satisfying']}/{v['samples_observed']}")


def d4_gap_does_not_end_the_run():
    """A SECOND consequence of the same fall-through, not named in the brief.

    `runnable = pending + claimed + staging` summed an unobserved sample to
    0+0+0, so a spell of failed ledger reads started the quiescence timer and
    could stop the sampler with 'run is over' while the run was alive. The gap
    must neither start nor clear that timer."""
    path = _static_saturated_ledger()
    tmp = Path(path).parent
    out, summ = str(tmp / "c2.tsv"), str(tmp / "v2.txt")
    # every loop read fails, and quiescence is 0.1s against a 0.5s run
    with failing_connect(set(range(2, 500))):
        rc, log = run_main(path, out, summ,
                           extra=["--run-id", "runGAP", "--quiesce-seconds", "0.1"])
    rows = open(out).read().splitlines()[1:]
    v = open(summ).read()
    check("D4-GAP-DOES-NOT-END-THE-RUN",
          "run is over" not in log
          and "max-seconds reached" in log
          and len(rows) >= 4
          and f"UNOBSERVED      : {len(rows)}" in v
          and f"OBSERVED        : 0" in v,
          f"{len(rows)} samples, all unobserved, sampler ran to max-seconds "
          f"instead of declaring the run over (rc={rc})")


# ──────────────────────────────────────────────────────────────────────────────
# E: TURNOVER IS MEASURED OVER THE QUALIFYING WINDOW  (Beta R2 §2)
# ──────────────────────────────────────────────────────────────────────────────

def e1_turnover_window_is_not_the_run():
    """The drain happens EARLY, while only 3 workers are active; the fleet then
    fills and holds static. Run-wide, pending fell 20 -> 10 and a naive reading
    calls that turnover. Inside the qualifying window nothing moved."""
    w25 = {f"w{i}" for i in range(25)}
    w3 = {"w0", "w1", "w2"}
    seq = [(w3, 20), (w3, 15), (w3, 10),          # drained, but fleet not full
           (w25, 10), (w25, 10), (w25, 10)]        # full, and nothing moves
    v = S.evaluate(samples_from(seq), 25, 2)
    run_wide_delta = seq[0][1] - seq[-1][1]
    check("E1-TURNOVER-WINDOW-IS-NOT-THE-RUN",
          v["satisfied"] is True
          and v["turnover_satisfied"] is False
          and v["turnover_window_samples"] == 3
          and v["turnover_pending_first"] == 10     # window start, not run start
          and v["turnover_pending_drained"] == 0
          and run_wide_delta == 10,                 # what the run-wide read says
          f"run-wide pending fell {run_wide_delta}, but within the qualifying "
          f"window ({v['turnover_window_samples']} samples) drained="
          f"{v['turnover_pending_drained']} -> NOT satisfied")


def e2_drain_during_an_occupancy_dip_is_not_credited():
    """Consumption must be paired with occupancy across the SAME samples. Here
    the queue drains 7 -> 1 precisely while occupancy dips to 3, which splits
    the run into two qualifying windows, neither of which contains the drain."""
    w25 = {f"w{i}" for i in range(25)}
    w3 = {"w0", "w1", "w2"}
    seq = [(w25, 7), (w25, 7), (w3, 4), (w25, 1), (w25, 1)]
    v = S.evaluate(samples_from(seq), 25, 2)
    check("E2-DIP-DRAIN-NOT-CREDITED",
          v["satisfied"] is True
          and v["qualifying_window_count"] == 2
          and v["turnover_satisfied"] is False
          and v["turnover_pending_drained"] == 0
          and v["turnover_window_min_active"] == 25,
          f"{v['qualifying_window_count']} qualifying windows, drain happened "
          f"between them -> turnover NOT satisfied")


def e3_stagewise_refill_still_counts():
    """Why the drain is summed STEP-WISE rather than read off the endpoints.

    A stage boundary inside the window replenishes the backlog: pending falls
    7 -> 3 (real consumption, under full occupancy) and then rises to 10 as the
    next stage's stripes are created. The endpoint delta is NEGATIVE and would
    report no turnover; the step-wise sum sees the four stripes that were
    actually consumed."""
    w25 = {f"w{i}" for i in range(25)}
    v = S.evaluate(samples_from([(w25, 7), (w25, 3), (w25, 10), (w25, 10)]), 25, 2)
    check("E3-STEPWISE-DRAIN-COUNTED",
          v["satisfied"] is True
          and v["turnover_satisfied"] is True
          and v["turnover_pending_drained"] == 4
          and v["turnover_pending_delta"] == -3      # endpoints say the opposite
          and v["turnover_transitions"] == 0,        # so drained is carrying it
          f"drained={v['turnover_pending_drained']} across steps while the "
          f"endpoint delta is {v['turnover_pending_delta']}")


# ──────────────────────────────────────────────────────────────────────────────
# B: TURNOVER IS EXISTENTIAL OVER ALL QUALIFYING WINDOWS  (Beta §§17-19)
# ──────────────────────────────────────────────────────────────────────────────

def _longest_only(v):
    """THE PRE-FIX AGGREGATION, reconstructed from the verdict's own per-window
    measurements: hand `_turnover` the single longest qualifying window, which
    is exactly what `evaluate` used to pass it (`qualifying = best`).

    Built from `windows_detail` rather than from a saved copy of the old source
    so the mutant cannot drift away from the code under test — it is the same
    aggregator, differing only in the set it quantifies over.
    """
    if not v["windows_detail"]:
        return S._turnover([])
    longest = max(v["windows_detail"], key=lambda w: (w["samples"], w["seconds"]))
    return S._turnover([longest])


def b1_shorter_window_turnover_is_credited():
    """THE ARM THIS ITEM EXISTS FOR (Beta §19-B1).

    The fleet saturates and holds 7 pending flat for five samples, dips, then
    saturates again and genuinely drains 7 -> 3. The LONGEST qualifying window
    is the static one; the shorter, later window is where the queue actually
    moved. Longest-only aggregation reports NO TURNOVER and publishes a run
    that demonstrably consumed queued work under full occupancy as unsaturated.

    The shorter window is not weaker evidence. Every step it counts is bracketed
    by two at-threshold samples, exactly as in the longest window — it is simply
    shorter. Turnover is a question of whether the queue EVER moved under full
    occupancy, so the aggregate is EXISTS, and the longest window being static
    is not evidence against a different window that moved.
    """
    w25, w3 = _w25(), {"w0", "w1", "w2"}
    seq = ([(w25, 7)] * 5           # window 1: LONGEST, and completely static
           + [(w3, 7)]              # dip -- breaks the run
           + [(w25, 7), (w25, 5), (w25, 3)])   # window 2: SHORTER, drains 4
    v = S.evaluate(samples_from(seq), 25, 2)
    text = S.render_summary(v, "runB1", "t0", "t1", "/x/db")
    wd = v["windows_detail"]
    mutant = _longest_only(v)

    check("DB1-SHORTER-WINDOW-TURNOVER-CREDITED",
          v["satisfied"] is True
          and v["qualifying_window_count"] == 2
          and v["longest_window_samples"] == 5        # the static one is longest
          and wd[0]["turnover"] is False              # ...and shows no turnover
          and wd[1]["samples"] == 3 and wd[1]["turnover"] is True
          and v["turnover_satisfied"] is True
          and v["turnover_witness_window"] == 2       # the shorter window
          and v["turnover_pending_drained"] == 4      # measured in the witness
          and S.overall_satisfied(v) is True
          and S.exit_code(v) == 0
          and f"{S.LABEL_OVERALL:<42}: SATISFIED" in text
          # RED-FIRST: the pre-fix aggregation over the longest window alone
          # calls this same run NOT SATISFIED.
          and mutant["turnover_satisfied"] is False,
          f"longest window ({v['longest_window_samples']} samples) static, "
          f"witness is window {v['turnover_witness_window']} "
          f"({wd[1]['samples']} samples, drained {wd[1]['pending_drained']}) "
          f"-> SATISFIED; longest-only mutant -> "
          f"{'SATISFIED' if mutant['turnover_satisfied'] else 'NOT SATISFIED'}")


def b2_drain_outside_occupancy_still_not_credited():
    """Beta §19-B2 — prove the step-wise-inside-window rule SURVIVES the change.

    Widening the aggregation from one window to all is the one change that could
    plausibly leak movement in from outside occupancy, so the property is
    re-proved against a MULTI-window fixture: pending falls 10 -> 2 precisely
    while occupancy dips to 3. Run-wide the queue drained 8. Inside window 1 it
    is flat at 10, inside window 2 flat at 2, and the drop belongs to neither
    because no consecutive pair lies inside a window and spans it.
    """
    w25, w3 = _w25(), {"w0", "w1", "w2"}
    seq = [(w25, 10), (w25, 10), (w3, 6), (w3, 2), (w25, 2), (w25, 2)]
    v = S.evaluate(samples_from(seq), 25, 2)
    run_wide = seq[0][1] - seq[-1][1]

    check("DB2-DIP-DRAIN-NOT-CREDITED-MULTIWINDOW",
          v["satisfied"] is True
          and v["qualifying_window_count"] == 2
          and all(w["turnover"] is False for w in v["windows_detail"])
          and all(w["pending_drained"] == 0 for w in v["windows_detail"])
          and v["turnover_satisfied"] is False
          and v["turnover_witness_window"] is None
          and S.overall_satisfied(v) is False
          and S.exit_code(v) == 3
          and run_wide == 8,
          f"run-wide pending fell {run_wide} during an occupancy dip; both "
          f"qualifying windows drained 0 -> existential aggregate still NOT "
          f"satisfied (rc={S.exit_code(v)})")


def b3_gap_forbids_borrowed_turnover():
    """Beta §19-B3 — no turnover may be inferred ACROSS an UNOBSERVED gap.

    A window broken by a gap is two windows, and the existential aggregate must
    not become a back door for the movement the gap conceals: pending is 10
    before it and 4 after, but nothing in the evidence file can say whether that
    drain happened under full occupancy or while the fleet stood empty. Neither
    window may borrow the other's endpoint.
    """
    w25 = _w25()
    built = samples_from([(w25, 10), (w25, 10), (w25, 10), (w25, 4), (w25, 4)])
    built[2] = gap(built[2])
    v = S.evaluate(built, 25, 2)
    # The control: the identical fixture with that sample OBSERVED is one
    # window, and the drain inside it is then genuinely creditable.
    v_cont = S.evaluate(samples_from(
        [(w25, 10), (w25, 10), (w25, 10), (w25, 4), (w25, 4)]), 25, 2)

    check("DB3-GAP-FORBIDS-BORROWED-TURNOVER",
          v["samples_unobserved"] == 1
          and v["qualifying_window_count"] == 2
          and all(w["pending_drained"] == 0 for w in v["windows_detail"])
          and v["turnover_satisfied"] is False
          and v["turnover_witness_window"] is None
          and v_cont["qualifying_window_count"] == 1      # clean control
          and v_cont["turnover_satisfied"] is True
          and v_cont["turnover_pending_drained"] == 6,
          f"gap splits 5 samples into {v['qualifying_window_count']} windows, "
          f"neither drains -> NOT satisfied; the same fixture without the gap "
          f"is 1 window draining {v_cont['turnover_pending_drained']}")


def b4_no_turnover_anywhere_is_still_not_satisfied():
    """Beta §19-B4 — THE CONTROL. Several qualifying windows, every one static.

    B1 widens what can satisfy verdict 2, and the risk a widening always carries
    is that it satisfies something. A run where the fleet repeatedly saturated
    and the queue never moved in ANY window must still fail, with the same exit
    code and the same authority line as before the change.
    """
    w25, w3 = _w25(), {"w0", "w1", "w2"}
    seq = ([(w25, 7)] * 3 + [(w3, 7)]
           + [(w25, 7)] * 3 + [(w3, 7)]
           + [(w25, 7)] * 2)
    v = S.evaluate(samples_from(seq), 25, 2)
    text = S.render_summary(v, "runB4", "t0", "t1", "/x/db")

    check("DB4-NO-TURNOVER-ANYWHERE-CONTROL",
          v["satisfied"] is True
          and v["qualifying_window_count"] == 3
          and all(w["turnover"] is False for w in v["windows_detail"])
          and v["turnover_satisfied"] is False
          and v["turnover_witness_window"] is None
          and v["turnover_windows_with_turnover"] == 0
          and v["turnover_basis_window"] == 1       # earliest, stated as such
          and S.overall_satisfied(v) is False
          and S.exit_code(v) == 3
          and f"{S.LABEL_OVERALL:<42}: NOT SATISFIED" in text
          and "TURNOVER WITNESS                         : NONE" in text
          and "WHY NOT (verdict 2)" in text,
          f"{v['qualifying_window_count']} qualifying windows, "
          f"{v['turnover_windows_with_turnover']} with turnover -> NOT "
          f"SATISFIED, no witness (rc={S.exit_code(v)})")


def b5_witness_is_deterministic_and_labelled():
    """Beta §19-B5 — the witness is CHOSEN, DOCUMENTED, LABELLED and STABLE.

    Three qualifying windows: window 1 static, windows 2 and 3 both holding
    turnover, and window 3 is deliberately the LONGEST. The witness must be
    window 2 -- the documented rule is EARLIEST-qualifying-window-with-turnover,
    so a run must never silently present the longest window as the basis, which
    is the substitution the defect made structural.

    Stability is checked by evaluating identical input twice and requiring the
    rendered evidence to be byte-identical: a witness picked by set iteration or
    by a max() over an unstable key would satisfy the same assertions once and
    not the next time.
    """
    w25, w3 = _w25(), {"w0", "w1", "w2"}
    seq = ([(w25, 7), (w25, 7)]                      # window 1: static
           + [(w3, 7)]
           + [(w25, 7), (w25, 5), (w25, 4)]          # window 2: drains 3
           + [(w3, 4)]
           + [(w25, 4), (w25, 3), (w25, 2), (w25, 1)])  # window 3: LONGEST, drains 3
    v = S.evaluate(samples_from(seq), 25, 2)
    text = S.render_summary(v, "runB5", "t0", "t1", "/x/db")
    again = S.render_summary(S.evaluate(samples_from(seq), 25, 2),
                             "runB5", "t0", "t1", "/x/db")
    wd = v["windows_detail"]
    longest_i = 1 + max(range(len(wd)), key=lambda i: (wd[i]["samples"],
                                                       wd[i]["seconds"]))

    check("DB5-WITNESS-DETERMINISTIC-AND-LABELLED",
          v["qualifying_window_count"] == 3
          and [w["turnover"] for w in wd] == [False, True, True]
          and longest_i == 3                       # the longest also has turnover
          and v["turnover_witness_window"] == 2    # ...and is NOT the witness
          and v["turnover_windows_with_turnover"] == 2
          and v["turnover_basis_window"] == 2      # basis IS the witness
          and v["turnover_window_samples"] == 3    # window 2's geometry, not 4
          and "EARLIEST" in v["turnover_witness_rule"]
          and "TURNOVER WITNESS                         : window 2" in text
          and text.count("<-- TURNOVER WITNESS") == 1
          and v["turnover_witness_rule"] in text   # the rule is self-described
          and "CONTEXT ONLY" in text               # longest labelled as context
          and again == text,                       # stable on identical input
          f"windows with turnover: 2 and {longest_i} (the longest); witness is "
          f"window {v['turnover_witness_window']} by the earliest rule, "
          f"labelled once, render stable across repeat evaluation")


# ──────────────────────────────────────────────────────────────────────────────
# F: THE EVIDENCE FILE STANDS ALONE  (Beta R2 §4)
# ──────────────────────────────────────────────────────────────────────────────

def f1_summary_is_self_describing():
    """Beta reads this file WITHOUT the report beside it."""
    w25 = {f"w{i}" for i in range(25)}
    built = samples_from([(w25, 7), (w25, 5), (w25, 3), (w25, 3)])
    built[3] = gap(built[3])
    v = S.evaluate(built, 25, 2)
    text = S.render_summary(v, "runA", "t0", "t1", "/x/db",
                            estab=S.summarize_estab(built), interval=2.0)

    required = {
        "threshold": "occupancy threshold : 25",
        "interval": "sample interval   : 2s",
        "total samples": "samples emitted   : 4",
        "observed": "OBSERVED        : 3",
        "unobserved": "UNOBSERVED      : 1",
        "window minimum": "window minimum    : 2 consecutive",
        "qualifying windows": "QUALIFYING",
        "window detail": "window 1: 3 samples",
        "predicate 1": "compute_active >= 25  AND  queued_pending >= 1",
        "predicate 2": "pending_drained > 0  OR  transitions > 0",
        "turnover window def": "QUALIFYING SIMULTANEITY WINDOW",
        "not the whole run": "NOT the whole run",
        # R3/S3: the combined authority line is a REQUIRED element. Without it
        # in this list a mutant deleting the line still passes the 17-element
        # check — the element list is what makes "self-describing" enforceable,
        # so anything the file must carry has to be named here.
        "combined authority": S.LABEL_OVERALL,
        "authority is the conjunction": "CONJUNCTION of the two verdicts",
        "verdict 1": S.LABEL_SIMULTANEITY,
        "verdict 2": S.LABEL_TURNOVER,
        "exit code": "EXIT CODE",
        "exit legend": "0 = both criteria satisfied",
        "gap rule": "BREAKS any",
    }
    missing = [k for k, needle in required.items() if needle not in text]
    check("F1-SUMMARY-IS-SELF-DESCRIBING", not missing,
          f"all {len(required)} required elements present"
          if not missing else f"MISSING: {missing}")


def f2_exit_code_matches_the_legend():
    """The legend the summary prints must be the mapping the process returns."""
    w25 = {f"w{i}" for i in range(25)}
    both = S.evaluate(samples_from([(w25, 7), (w25, 4), (w25, 2)]), 25, 2)
    turn = S.evaluate(samples_from([(w25, 7)] * 3), 25, 2)
    neither = S.evaluate(samples_from([({"w0"}, 7)] * 3), 25, 2)
    check("F2-EXIT-CODE-MATCHES-LEGEND",
          (S.exit_code(both), S.exit_code(turn), S.exit_code(neither)) == (0, 3, 2)
          and "EXIT CODE" + " " * 33 + ": 3" in
              S.render_summary(turn, "r", "t0", "t1", "/x/db"),
          f"both={S.exit_code(both)} turnover-only={S.exit_code(turn)} "
          f"neither={S.exit_code(neither)}; summary prints the same code")


# ──────────────────────────────────────────────────────────────────────────────
# S3: THE COMBINED SATURATION AUTHORITY LINE  (Beta R3)
# ──────────────────────────────────────────────────────────────────────────────

def _w25():
    return {f"w{i}" for i in range(25)}


def _both_and_simultaneity_only():
    """Two verdict dicts from the REAL `evaluate`, not hand-built.

    `both`: occupancy held AND the queue drained -> 1 yes, 2 yes.
    `sim_only`: occupancy held but the queue never moved -> 1 yes, 2 no. That
    second one is the case the authority line exists to catch: a run that looks
    saturated on the headline number and never proved consumption.
    """
    w = _w25()
    both = S.evaluate(samples_from([(w, 7), (w, 4), (w, 2)]), 25, 2)
    sim_only = S.evaluate(samples_from([(w, 7)] * 3), 25, 2)
    return both, sim_only


def s3_authority_line_is_the_conjunction():
    """The artifact carries the Gate-12 result itself.

    Before this, the combined semantics lived only in `exit_code()` — and
    `gate12_launch.sh` never reads that exit code, so the conclusion existed
    nowhere a reader of the evidence file could see it.
    """
    both, sim_only = _both_and_simultaneity_only()
    t_both = S.render_summary(both, "rB", "t0", "t1", "/x/db")
    t_sim = S.render_summary(sim_only, "rS", "t0", "t1", "/x/db")

    line_both = f"{S.LABEL_OVERALL:<42}: SATISFIED"
    line_sim = f"{S.LABEL_OVERALL:<42}: NOT SATISFIED"

    # The sub-verdicts must SURVIVE underneath, unchanged — Beta keeps them as
    # the diagnostic that says WHICH criterion failed.
    subs_kept = (f"{S.LABEL_SIMULTANEITY:<42}: SATISFIED" in t_sim
                 and f"{S.LABEL_TURNOVER:<42}: NOT SATISFIED" in t_sim)
    # ...and the authority line must sit ABOVE them, not after.
    above = (t_sim.index(S.LABEL_OVERALL) < t_sim.index(S.LABEL_SIMULTANEITY)
             < t_sim.index(S.LABEL_TURNOVER))

    check("S3-AUTHORITY-LINE-IS-CONJUNCTION",
          line_both in t_both and line_sim in t_sim and subs_kept and above
          and S.overall_satisfied(both) and not S.overall_satisfied(sim_only),
          "yes/yes -> SATISFIED; yes/no -> NOT SATISFIED; both sub-verdicts "
          "retained beneath the authority line")


def s3_mutant_or_instead_of_and():
    """MUTATION: compute the authority line with `or` instead of `and`.

    The mutant is the exact collapse Beta forbids — a run that proved occupancy
    but never proved the queue was consumed would publish itself as SATISFIED.
    """
    _both, sim_only = _both_and_simultaneity_only()
    fixed = S.render_summary(sim_only, "rS", "t0", "t1", "/x/db")
    real = S.overall_satisfied
    try:
        S.overall_satisfied = lambda v: bool(
            v["satisfied"] or v["turnover_satisfied"])
        mutant = S.render_summary(sim_only, "rS", "t0", "t1", "/x/db")
    finally:
        S.overall_satisfied = real

    line_sat = f"{S.LABEL_OVERALL:<42}: SATISFIED"
    line_not = f"{S.LABEL_OVERALL:<42}: NOT SATISFIED"
    check("S3-MUTANT-OR-COLLAPSES-VERDICT",
          line_not in fixed and line_sat in mutant,
          "`or` mutant publishes a turnover-failed run as SATISFIED; "
          "the `and` implementation reports NOT SATISFIED")


# ──────────────────────────────────────────────────────────────────────────────
# S4: DURABLE SIMULTANEOUS WORKER IDENTITIES IN THE TSV  (Beta R3)
# ──────────────────────────────────────────────────────────────────────────────

def s4_identities_persist_and_gaps_are_not_empty():
    """END TO END through main(). Two of the three rendering cases at once:
    observed rows must carry the identities that make `compute_active`
    auditable, and unobserved rows must NOT carry `[]`."""
    path = _static_saturated_ledger()
    tmp = Path(path).parent
    out, summ = str(tmp / "s4.tsv"), str(tmp / "s4v.txt")
    with failing_connect({3, 4}):
        rc, _log = run_main(path, out, summ, extra=["--run-id", "runGAP"])

    rows = [l.rstrip("\n").split("\t")
            for l in open(out).read().splitlines()[1:]]
    cols = {c: i for i, c in enumerate(S.TSV_COLUMNS)}
    i_aw, i_ca = cols["active_workers_json"], cols["compute_active"]
    obs = [r for r in rows if r[cols["obs_status"]] == S.OBS_OBSERVED]
    unobs = [r for r in rows if r[cols["obs_status"]] == S.OBS_UNOBSERVED]

    expected = sorted(f"w{i}" for i in range(25))
    parsed = [json.loads(r[i_aw]) for r in obs]
    # the identities are all 25, and SORTED as stored (not merely as a set)
    ids_ok = all(p == expected for p in parsed)
    # THE INVARIANT, checked from the persisted bytes rather than in-process
    invariant_ok = all(len(p) == int(r[i_ca]) for p, r in zip(parsed, obs))
    # the gap carries the marker and — the actual defect — never `[]`
    gaps_marked = all(r[i_aw] == S.OBS_UNOBSERVED for r in unobs)
    no_empty_array = all(r[i_aw] != "[]" for r in unobs)

    check("S4-IDENTITIES-PERSIST-GAPS-ARE-NOT-EMPTY",
          len(obs) >= 2 and len(unobs) == 2 and ids_ok and invariant_ok
          and gaps_marked and no_empty_array,
          f"{len(obs)} observed rows carry 25 sorted ids with "
          f"len==compute_active; {len(unobs)} gaps carry "
          f"{S.OBS_UNOBSERVED}, never '[]' (rc={rc})")


def s4_observed_zero_is_a_real_empty_array():
    """THE DISTINCTION THAT IS THE POINT. A ledger that was read successfully
    and holds no run of ours is an observation OF NOTHING: `[]`. That is a
    different claim from a read that never happened, and the file must be able
    to say which one it is."""
    empty = build_ledger({"someone_elses_run": [(S.ST_DONE, None, 3)]},
                         created_at=1.0, wal=True)     # created before start
    tmp = Path(empty).parent
    out, summ = str(tmp / "z.tsv"), str(tmp / "zv.txt")
    rc, _log = run_main(empty, out, summ)              # no --run-id: none latches

    rows = [l.rstrip("\n").split("\t")
            for l in open(out).read().splitlines()[1:]]
    cols = {c: i for i, c in enumerate(S.TSV_COLUMNS)}
    i_aw = cols["active_workers_json"]

    all_observed = all(r[cols["obs_status"]] == S.OBS_OBSERVED for r in rows)
    all_empty = all(r[i_aw] == "[]" for r in rows)
    zero_count = all(r[cols["compute_active"]] == "0" for r in rows)

    check("S4-OBSERVED-ZERO-IS-A-REAL-EMPTY-ARRAY",
          rows and all_observed and all_empty and zero_count,
          f"{len(rows)} no-run-yet rows: obs_status=OBSERVED, "
          f"active_workers_json='[]', compute_active=0 — an observed zero, "
          f"distinguishable from the {S.OBS_UNOBSERVED} marker (rc={rc})")


def s4_mutant_unconditional_serialization():
    """MUTATION: serialize the set unconditionally instead of keying off
    `obs_status`.

    This is not a hypothetical mutant. `unobserved_row` really does seed
    `active_workers: set()`, so the naive renderer emits `[]` on exactly the
    samples where `[]` is false — a failed read presented as an observed
    zero-worker instant.
    """
    g = S.unobserved_row(1000.0, "T+0s", "OperationalError:database is locked")
    fixed = S.format_tsv_row(g, "runX", None).rstrip("\n").split("\t")
    i_aw = S.TSV_COLUMNS.index("active_workers_json")

    real = S.render_active_workers
    try:
        S.render_active_workers = lambda row: json.dumps(
            sorted(row.get("active_workers") or ()), separators=(",", ":"))
        mutant = S.format_tsv_row(g, "runX", None).rstrip("\n").split("\t")
    finally:
        S.render_active_workers = real

    check("S4-MUTANT-UNCONDITIONAL-SERIALIZATION",
          fixed[i_aw] == S.OBS_UNOBSERVED and mutant[i_aw] == "[]",
          f"mutant renders a failed read as '[]' (an observed zero-worker "
          f"instant); the obs_status-keyed renderer emits {S.OBS_UNOBSERVED}")


def s4_invariant_is_enforced_at_the_render_site():
    """The invariant must be ENFORCED, not merely documented: a row whose count
    and identities disagree must not be able to reach the evidence file."""
    bad = {"epoch": 1000.0, "ts_iso": "T+0s", "obs_status": S.OBS_OBSERVED,
           "obs_reason": None, "active_workers": {"w0", "w1"},
           "compute_active": 25, "queued_pending": 7, "claimed_rows": 25,
           "staging": 0, "done": 0, "cancelled": 0, "failed": 0}
    raised = ""
    try:
        S.format_tsv_row(bad, "runX", True)
    except AssertionError as e:
        raised = str(e)
    # control: the same row with a consistent count renders fine
    ok_row = dict(bad, compute_active=2, claimed_rows=2)
    rendered = S.format_tsv_row(ok_row, "runX", True)

    check("S4-INVARIANT-ENFORCED-AT-RENDER",
          "disagree" in raised and "compute_active=25" in raised
          and '["w0","w1"]' in rendered,
          "2 identities vs compute_active=25 is refused at the render site; "
          "the consistent control renders")


# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("GATE-12 CONCURRENCY SAMPLER — post-F1 state model")
    print("=" * 70)
    print("\n-- state model --")
    g1_g2_g3_state_model()
    print("\n-- verdict --")
    g4_full_but_empty_queue()
    g5_union_across_instants()
    g6_clean_control()
    g6b_single_sample_blip()
    print("\n-- turnover (R1 §5): a SECOND, separate verdict --")
    c1_static_queue_no_turnover()
    c2_draining_queue_both()
    c3_union_only_neither()
    c4_transitions_alone_suffice()
    c5_verdicts_are_not_collapsed()
    print("\n-- scoping and safety --")
    g7_run_scoping()
    g8_latch_after_start()
    g9_connection_is_readonly()
    g10_production_db_refused()
    print("\n-- atomicity (R1 §3): one sample is one snapshot --")
    a1_a3_snapshot_is_atomic()
    a2_mutation_two_autocommit_reads()
    print("\n-- ESTAB honesty (R1 §4): unavailable is never 0 --")
    b1_b5_estab_outcomes()
    b6_tsv_never_renders_zero()
    b7_verdict_unaffected_by_estab()
    print("\n-- unobserved ledger reads (R2 §1): a gap is never a zero --")
    d1_injected_read_failure_is_unobserved()
    d2_gap_breaks_the_window()
    d3_unobserved_never_enters_as_zero()
    d4_gap_does_not_end_the_run()
    print("\n-- turnover window (R2 §2): the qualifying window, not the run --")
    e1_turnover_window_is_not_the_run()
    e2_drain_during_an_occupancy_dip_is_not_credited()
    e3_stagewise_refill_still_counts()
    print("\n-- defect B (§§17-19): turnover is EXISTENTIAL over all windows --")
    b1_shorter_window_turnover_is_credited()
    b2_drain_outside_occupancy_still_not_credited()
    b3_gap_forbids_borrowed_turnover()
    b4_no_turnover_anywhere_is_still_not_satisfied()
    b5_witness_is_deterministic_and_labelled()
    print("\n-- self-describing evidence (R2 §4) --")
    f1_summary_is_self_describing()
    f2_exit_code_matches_the_legend()
    print("\n-- S3 (R3): the combined saturation authority line --")
    s3_authority_line_is_the_conjunction()
    s3_mutant_or_instead_of_and()
    print("\n-- S4 (R3): durable simultaneous worker identities --")
    s4_identities_persist_and_gaps_are_not_empty()
    s4_observed_zero_is_a_real_empty_array()
    s4_invariant_is_enforced_at_the_render_site()
    s4_mutant_unconditional_serialization()
    print("\n-- mutation --")
    m1_legacy_query_reds_g1()

    passed = sum(1 for _, ok in _RESULTS if ok)
    total = len(_RESULTS)
    print("=" * 70)
    print(f"{passed}/{total} checks green")
    if passed != total:
        print("FAILURES: " + ", ".join(n for n, ok in _RESULTS if not ok))
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
