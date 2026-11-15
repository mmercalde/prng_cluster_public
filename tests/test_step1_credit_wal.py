#!/usr/bin/env python3
"""STEP1-CREDIT-WAL — the S7 evidence reader on a ledger whose `-wal` is non-empty.

Governing brief: docs/CLAUDE_CODE_INSTRUCTIONS_STEP1_CREDIT_WAL_20261115.md §4.
Run 4 attempt 1 (2026-11-15) swept cleanly and was refused credit because the miner
ledger's `-wal` was non-empty: the gate12 sampler (a `mode=ro` reader) was connected when
the coordinator's last writing connection closed, and a read-only last closer can neither
checkpoint nor delete the WAL.

Fixture (shared): a WAL-mode ledger built with the REAL MinerLedger API — one run, four
phases x N stripes — while a reader opened with the REAL sampler's `connect_ro` stays
connected across the writer's close. Afterwards `-wal` is non-empty and every stripe row
lives only in the WAL (the main file is stale).

  G-RED            base reader (HEAD c47680a): no evidence, UNAVAILABLE printed.   [red first]
  G-GREEN          new reader: evidence equals the committed rows, all 4 phases credited
                   by credit_decision; the ledger directory is byte-identical (names,
                   sizes, sha256, mtime_ns) and the scratch copy is removed.
  G-NOSHM          `-wal` present, `-shm` absent: credited, and no `-shm` is created.
  G-STALE          the main file alone gives the wrong answer (immutable sees no rows);
                   the new reader returns the committed rows.
  R-PHASE-MISSING  one required phase has no rows                       -> no credit
  R-NOT-DONE       one stripe left `claimed`                             -> no credit
  R-LIVE-WRITER    successor of part3 G-EVID-WAL: complete baseline checkpointed into the
                   main file (immutable view = stale positive), then a still-connected
                   writer commits a phase-1 stripe back to `staging`     -> no credit
  R-GAP            a hole in one phase's tiling                          -> no credit
  R-NOT-COMMITTED  miner_result committed=False                          -> no evidence
  R-UNREADABLE     corrupt database bytes                                -> no evidence
  R-JOURNAL        non-empty `-journal`                                  -> no evidence
  R-MOVING         the source `-wal` grows during the copy               -> no evidence
Mutants (each must turn its named gate RED):
  M1 base reader                          -> G-GREEN
  M2 plain `mode=ro` on the original      -> G-GREEN (directory changed)
  M3 `immutable=1` on the original        -> G-STALE
  M4 copy without the stability check     -> R-MOVING
  M5 `immutable=1` on the original        -> R-LIVE-WRITER (credits phase 1)
Every gate terminates PASS | FAIL; only PASS accepts.

Optional: `--frozen <ledger dir>` copies a frozen ledger (e.g. the Run 4 attempt-1
bundle) into suite scratch and prints what the new reader returns for `--frozen-run`.
That prints evidence only; it credits nothing and is not a gate.
"""
import argparse
import ast
import contextlib
import hashlib
import importlib
import importlib.util
import io
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
SCRATCH = os.path.join(ROOT, "data", "step1_credit_wal", "suite_tmp")
os.makedirs(SCRATCH, exist_ok=True)
BASE = "c47680a24a5634445b8548f5db00b0150feb17f4"
I_MOD = "window_optimizer_integration_final"
RUN = "credwal_t1_0000"
PHASES = (1, 2, 3, 4)
START, STRIPE, N = 2 ** 31, 1 << 20, 8
LO, HI = START, START + N * STRIPE
RESULTS, MUTANTS = [], []


class Fail(AssertionError):
    pass


def ok_(cond, msg):
    if not cond:
        raise Fail(msg)


@contextlib.contextmanager
def tmpdir(prefix):
    d = tempfile.mkdtemp(prefix=prefix, dir=SCRATCH)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


@contextlib.contextmanager
def private_tempdir():
    """Point the reader's tempfile.mkdtemp() at a watched directory, so a gate can show
    the scratch copy was removed."""
    d = tempfile.mkdtemp(prefix="tmproot_", dir=SCRATCH)
    old = tempfile.tempdir
    tempfile.tempdir = d
    try:
        yield d
    finally:
        tempfile.tempdir = old
        shutil.rmtree(d, ignore_errors=True)


def run_gate(name, fn):
    try:
        RESULTS.append((name, True, fn() or ""))
    except BaseException as e:   # noqa: BLE001
        RESULTS.append((name, False, f"{type(e).__name__}: {e}"[:1200]))


def _sampler():
    spec = importlib.util.spec_from_file_location(
        "gate12_concurrency_sampler", os.path.join(ROOT, "scripts",
                                                   "gate12_concurrency_sampler.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def listing(d):
    out = {}
    for n in sorted(os.listdir(d)):
        p = os.path.join(d, n)
        st = os.stat(p)
        with open(p, "rb") as f:
            out[n] = (st.st_size, st.st_mtime_ns, hashlib.sha256(f.read()).hexdigest())
    return out


def make_fixture(d, *, phases=PHASES, leave=None, gap=None):
    """Real MinerLedger writes with a real sampler `connect_ro` reader connected across
    the writer's close. `leave=(phase, i, state)` leaves one stripe in that state;
    `gap=(phase, i)` omits stripe i of that phase. Returns the ledger path."""
    from miner.range_miner_coordinator import MinerLedger, ST_DONE
    path = os.path.join(d, "miner_ledger.db")
    led = MinerLedger(path)
    smp = _sampler()
    reader = smp.connect_ro(path)
    with smp.read_snapshot(reader):
        reader.execute("SELECT COUNT(*) FROM stripes").fetchone()
    led.create_trial(RUN, 1, now=1.0)
    for ph in phases:
        for i in range(N):
            if gap == (ph, i):
                continue
            sid = f"{RUN}__st{ph - 1}_s{i}"
            led.add_stripe(RUN, sid, START + i * STRIPE, STRIPE, "java_lcg", ph, now=1.0)
            st = leave[2] if leave and (ph, i) == leave[:2] else ST_DONE
            led.set_stripe_state(RUN, sid, st)
    del led                       # the writer closes while the reader is connected
    with smp.read_snapshot(reader):
        reader.execute("SELECT COUNT(*) FROM stripes").fetchone()
    reader.close()                # the read-only reader closes LAST
    ok_(os.path.exists(path + "-wal") and os.path.getsize(path + "-wal") > 0,
        f"fixture: -wal is not non-empty ({sorted(os.listdir(d))})")
    return path


def reader_fn():
    return importlib.import_module(I_MOD)._step1_miner_sweep_evidence


def call(reader, d, committed=True):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ev = reader({"run_id": RUN, "committed": committed},
                    types.SimpleNamespace(staging_dir=d, miner_output_dir=None))
    return ev, buf.getvalue()


def credit(ev):
    from utils.step1_search_ledger import credit_decision
    return credit_decision(requested_interval=(LO, HI), required_phases=list(PHASES),
                           evidence_by_trial={} if ev is None else {0: ev},
                           trial_states={0: "COMPLETE"})


def committed_rows_independent(d):
    """The committed rows, read independently: copy the WHOLE directory elsewhere and
    open the copy normally (SQLite recovers the WAL)."""
    with tmpdir("indep_") as e:
        for n in os.listdir(d):
            shutil.copyfile(os.path.join(d, n), os.path.join(e, n))
        c = sqlite3.connect(os.path.join(e, "miner_ledger.db"))
        try:
            return c.execute("SELECT phase, seed_start, seed_count, state FROM stripes "
                             "WHERE run_id=? ORDER BY phase, seed_start", (RUN,)).fetchall()
        finally:
            c.close()


# ===========================================================================
def g_red(reader=None):
    reader = reader or _base_reader()
    with tmpdir("red_") as d:
        make_fixture(d)
        ev, out = call(reader, d)
        ok_(ev is None, f"base reader returned evidence: {ev}")
        ok_("UNAVAILABLE" in out, f"no UNAVAILABLE printed: {out!r}")
        credited, code = credit(ev)[0:2]
        ok_(not credited, "credit_decision credited with no evidence")
    return f"base reader: None, UNAVAILABLE, credit_decision -> {code}"


def g_green(reader=None):
    reader = reader or reader_fn()
    with tmpdir("green_") as d, private_tempdir() as tr:
        make_fixture(d)
        expect = committed_rows_independent(d)
        ok_(len(expect) == len(PHASES) * N and all(r[3] == "done" for r in expect),
            f"fixture: independent committed rows wrong: {expect}")
        before = listing(d)
        ev, out = call(reader, d)
        after = listing(d)
        ok_(ev is not None, f"no evidence: {out!r}")
        want = {str(p): [LO, HI] for p in PHASES}
        ok_(ev["swept_intervals"] == want, f"evidence {ev['swept_intervals']} != {want}")
        cr = credit(ev)
        ok_(cr[0], f"credit_decision refused: {cr}")
        ok_(after == before, f"the read changed the ledger directory:\n before {before}\n"
                             f" after  {after}")
        ok_(os.listdir(tr) == [], f"scratch copy left behind: {os.listdir(tr)}")
    return (f"{len(expect)} committed rows -> {sorted(ev['swept_intervals'])} all "
            f"[{LO},{HI}); credited; directory byte-identical {sorted(after)}; scratch removed")


def g_noshm(reader=None):
    reader = reader or reader_fn()
    with tmpdir("noshm_") as d, private_tempdir():
        path = make_fixture(d)
        if os.path.exists(path + "-shm"):
            os.remove(path + "-shm")
        before = listing(d)
        ev, out = call(reader, d)
        after = listing(d)
        ok_(ev is not None and credit(ev)[0], f"not credited: {ev} {out!r}")
        ok_(after == before, f"directory changed: {sorted(before)} -> {sorted(after)}")
    return f"-shm absent: credited; directory unchanged {sorted(after)}"


def g_stale(reader=None):
    reader = reader or reader_fn()
    with tmpdir("stale_") as d, private_tempdir():
        path = make_fixture(d)
        im = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
        try:
            n_im = im.execute("SELECT COUNT(*) FROM stripes WHERE run_id=?",
                              (RUN,)).fetchone()[0]
        finally:
            im.close()
        ok_(n_im == 0, f"fixture: main file is not stale (immutable sees {n_im} rows)")
        ev, out = call(reader, d)
        ok_(ev is not None and ev["swept_intervals"] == {str(p): [LO, HI] for p in PHASES},
            f"new reader did not return the committed rows: {ev} {out!r}")
    return f"immutable sees {n_im} rows; new reader returns all {len(PHASES)} phases"


def _refused(d, ev, out, expect_none=False):
    ok_(not credit(ev)[0], f"CREDITED: {ev}")
    if expect_none:
        ok_(ev is None, f"evidence returned: {ev}")
    return credit(ev)[1:3]


def r_phase_missing():
    with tmpdir("rpm_") as d, private_tempdir():
        make_fixture(d, phases=(1, 2, 3))
        ev, out = call(reader_fn(), d)
        return "refused: %s %s" % _refused(d, ev, out)


def r_not_done():
    with tmpdir("rnd_") as d, private_tempdir():
        make_fixture(d, leave=(2, 3, "claimed"))
        ev, out = call(reader_fn(), d)
        ok_(ev is not None and "2" not in ev["swept_intervals"], f"phase 2 credited: {ev}")
        return "refused: %s %s" % _refused(d, ev, out)


def r_live_writer(reader=None):
    """Successor of part3 G-EVID-WAL, with its stale-positive control (Beta X-STEP1-
    CREDIT-WAL R1 Q7). The COMPLETE baseline is first checkpointed into the main file, so
    an immutable read of the original sees every phase done. Then a still-connected
    writer commits stripe s0 of phase 1 back to `staging`, which stays only in the WAL.
    An immutable read would wrongly credit phase 1; the reader must not."""
    reader = reader or reader_fn()
    with tmpdir("rlw_") as d, private_tempdir():
        path = make_fixture(d)
        c = sqlite3.connect(path)                 # fixture copy only: settle the baseline
        try:
            c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            c.close()
        w = sqlite3.connect(path)
        try:
            w.execute("PRAGMA journal_mode=WAL")
            w.execute("PRAGMA wal_autocheckpoint=0")
            w.execute("UPDATE stripes SET state='staging' WHERE run_id=? AND phase=1 "
                      "AND stripe_id LIKE ?", (RUN, "%_s0"))
            w.commit()
            ok_(os.path.getsize(path + "-wal") > 0, "fixture: -wal is empty")
            im = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
            try:
                n_done = im.execute("SELECT COUNT(*) FROM stripes WHERE run_id=? AND "
                                    "phase=1 AND state='done'", (RUN,)).fetchone()[0]
            finally:
                im.close()
            ok_(n_done == N, f"fixture: immutable view is not the stale positive "
                             f"({n_done}/{N} phase-1 stripes done)")
            ev, out = call(reader, d)
        finally:
            w.close()
        ok_(ev is not None, f"no evidence at all (expected phases 2-4): {out!r}")
        ok_("1" not in ev["swept_intervals"], f"phase 1 credited: {ev}")
        ok_(sorted(ev["swept_intervals"]) == ["2", "3", "4"],
            f"phases other than 1 not read: {ev}")
        why = _refused(d, ev, out)
    return (f"immutable view: phase 1 {n_done}/{N} done (stale positive); reader: "
            f"phases {sorted(ev['swept_intervals'])}; refused: {why[0]} {why[1]}")


def r_gap():
    with tmpdir("rgap_") as d, private_tempdir():
        make_fixture(d, gap=(3, 4))
        ev, out = call(reader_fn(), d)
        ok_(ev is not None and "3" not in ev["swept_intervals"], f"phase 3 credited: {ev}")
        return "refused: %s %s" % _refused(d, ev, out)


def r_not_committed():
    with tmpdir("rnc_") as d, private_tempdir():
        make_fixture(d)
        ev, out = call(reader_fn(), d, committed=False)
        return "refused: %s %s" % _refused(d, ev, out, expect_none=True)


def r_unreadable():
    with tmpdir("run_") as d, private_tempdir() as tr:
        path = make_fixture(d)
        with open(path, "r+b") as f:
            f.write(b"\x00" * 4096)          # destroy the header page
        ev, out = call(reader_fn(), d)
        ok_("unreadable" in out, f"no 'unreadable' reason: {out!r}")
        ok_(os.listdir(tr) == [], f"scratch copy left behind: {os.listdir(tr)}")
        return "refused: %s %s" % _refused(d, ev, out, expect_none=True)


def r_journal():
    with tmpdir("rj_") as d, private_tempdir():
        path = make_fixture(d)
        with open(path + "-journal", "wb") as f:
            f.write(b"\x01" * 512)
        ev, out = call(reader_fn(), d)
        ok_("UNAVAILABLE" in out, f"no UNAVAILABLE: {out!r}")
        return "refused: %s %s" % _refused(d, ev, out, expect_none=True)


def r_moving(reader=None):
    reader = reader or reader_fn()
    with tmpdir("rmv_") as d, private_tempdir() as tr:
        path = make_fixture(d)
        real = shutil.copyfile

        def copy_then_grow(src, dst, *a, **k):
            r = real(src, dst, *a, **k)
            if src.endswith("-wal"):
                with open(src, "ab") as f:      # a writer appended after the copy
                    f.write(b"\x00" * 4120)
            return r
        shutil.copyfile = copy_then_grow
        try:
            ev, out = call(reader, d)
        finally:
            shutil.copyfile = real
        ok_(ev is None, f"evidence returned from a moving source: {ev}")
        ok_("changed during the evidence copy" in out, f"no stability reason: {out!r}")
        ok_(os.listdir(tr) == [], f"scratch copy left behind: {os.listdir(tr)}")
    return "source -wal grew during copy -> None, reason printed, scratch removed"


# ===========================================================================
def _base_reader():
    src = subprocess.check_output(["git", "-C", ROOT, "show",
                                   f"{BASE}:window_optimizer_integration_final.py"]).decode()
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)
                and n.name == "_step1_miner_sweep_evidence")
    lines = src.splitlines(keepends=True)[node.lineno - 1:node.end_lineno]
    mod = importlib.import_module(I_MOD)
    ns = {}
    exec(compile(textwrap.dedent("".join(lines)), "<base _step1_miner_sweep_evidence>",
                 "exec"), mod.__dict__, ns)
    return ns["_step1_miner_sweep_evidence"]


@contextlib.contextmanager
def patched_copy_reader(fn):
    mod = importlib.import_module(I_MOD)
    real = mod._step1_ledger_rows_via_copy
    mod._step1_ledger_rows_via_copy = fn
    try:
        yield
    finally:
        mod._step1_ledger_rows_via_copy = real


def _q(uri):
    def f(path, run_id):
        c = sqlite3.connect(uri.format(path=path), uri=True)
        try:
            return c.execute("SELECT phase, seed_start, seed_count, state FROM stripes "
                             "WHERE run_id=? ORDER BY phase, seed_start", (run_id,)).fetchall()
        finally:
            c.close()
    return f


def _no_stability_copy(path, run_id):
    scratch = tempfile.mkdtemp(prefix="step1_credit_evidence_")
    try:
        for p in (path, path + "-wal"):
            shutil.copyfile(p, os.path.join(scratch, os.path.basename(p)))
        return _q("file:{path}?mode=ro")(os.path.join(scratch, os.path.basename(path)),
                                         run_id)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _mutant(label, fn):
    try:
        det = fn()
        MUTANTS.append((label, False, f"gate stayed GREEN under the mutant: {det}"))
    except BaseException as e:   # noqa: BLE001
        MUTANTS.append((label, True, f"RED: {type(e).__name__}: {str(e)[:300]}"))


def _under(patch, gate):
    def f():
        with patched_copy_reader(patch):
            return gate()
    return f


GATES = [("G-RED", g_red), ("G-GREEN", g_green), ("G-NOSHM", g_noshm), ("G-STALE", g_stale),
         ("R-PHASE-MISSING", r_phase_missing), ("R-NOT-DONE", r_not_done),
         ("R-LIVE-WRITER", r_live_writer), ("R-GAP", r_gap),
         ("R-NOT-COMMITTED", r_not_committed), ("R-UNREADABLE", r_unreadable),
         ("R-JOURNAL", r_journal), ("R-MOVING", r_moving)]


def frozen(ledger_dir, run_id):
    global RUN
    with tmpdir("frozen_") as d, private_tempdir() as tr:
        for n in os.listdir(ledger_dir):
            shutil.copyfile(os.path.join(ledger_dir, n), os.path.join(d, n))
        before = listing(d)
        RUN = run_id
        ev, out = call(reader_fn(), d)
        after = listing(d)
        print(out.rstrip())
        print(f"FROZEN copy of {ledger_dir}, run {run_id}:")
        print(f"  files before: { {k: v[0] for k, v in before.items()} }")
        print(f"  evidence: {ev}")
        print(f"  copy directory unchanged by the read: {after == before}; "
              f"scratch removed: {os.listdir(tr) == []}")
        return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", default=None)
    ap.add_argument("--frozen-run", default="distributed_config_t1_57dc50fa")
    a = ap.parse_args()
    if a.frozen:
        frozen(a.frozen, a.frozen_run)
        return 0
    print("=" * 78)
    print("STEP1-CREDIT-WAL — S7 evidence reader with a non-empty ledger -wal")
    print("=" * 78)
    print(f"sqlite {sqlite3.sqlite_version}, python {sys.version.split()[0]}")
    for name, fn in GATES:
        run_gate(name, fn)
        n, ok, det = RESULTS[-1]
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}: {det}", flush=True)
    print("MUTANTS (each must turn its gate RED)")
    _mutant("M1 base reader (refuses any non-empty -wal) -> G-GREEN",
            lambda: g_green(_base_reader()))
    _mutant("M2 plain mode=ro on the original -> G-GREEN",
            _under(_q("file:{path}?mode=ro"), g_green))
    _mutant("M3 immutable=1 on the original -> G-STALE",
            _under(_q("file:{path}?mode=ro&immutable=1"), g_stale))
    _mutant("M4 copy without the stability check -> R-MOVING",
            _under(_no_stability_copy, r_moving))
    _mutant("M5 immutable=1 on the original -> R-LIVE-WRITER (stale positive)",
            _under(_q("file:{path}?mode=ro&immutable=1"), r_live_writer))
    for lab, det, txt in MUTANTS:
        print(f"  [{'RED (detected)' if det else 'NOT DETECTED'}] {lab}: {txt}", flush=True)
    g_ok = sum(1 for _, ok, _ in RESULTS if ok)
    m_ok = sum(1 for _, det, _ in MUTANTS if det)
    print(f"GATES {g_ok}/{len(RESULTS)}  MUTANTS {m_ok}/{len(MUTANTS)} detected")
    for n, ok, d in RESULTS:
        if not ok:
            print(f"--- FAIL {n}\n{d}")
    ok = g_ok == len(RESULTS) and m_ok == len(MUTANTS)
    print(f"COMPLETION SENTINEL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
