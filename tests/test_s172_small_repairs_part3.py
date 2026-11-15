#!/usr/bin/env python3
"""S172 — two small repairs before Run 4 (brief Part 3).

Governing brief: docs/CLAUDE_CODE_INSTRUCTIONS_MINER_ADMISSION_STALL_20261114.md §3.

  G-L5            a FRESH process (window_optimizer not imported), one real run_step(1)
                  on the pinned Run-4 path with dispatch intercepted: WATCHER's
                  `EXEC CMD` and `[S172-PIN] step1_pin_source` records are both
                  captured, window_optimizer WAS imported during run_step (so the path
                  L-5 names was exercised), and the root level is unchanged.
  G-EVID-OPEN     the S7 evidence reader on a throwaway WAL-mode miner ledger with no
                  sidecars: it returns the swept interval, and the directory beside the
                  ledger is unchanged afterwards (no `-wal`, no `-shm`, no other file).
  G-EVID-WAL      the same reader with a NON-EMPTY `-wal` (a writer holding an
                  uncheckpointed write): never a stale immutable read — the phase the
                  writer un-did is not credited. Since the owner ruling of 2026-11-15
                  (docs/CLAUDE_CODE_REPORT_STEP1_CREDIT_WAL.md) the committed state is
                  read (phase 2 credited) and the directory beside the ledger is
                  unchanged, instead of a blanket UNAVAILABLE.
Mutants: each repair reverted to its base (7b9184a) form must red its gate.
Every gate terminates PASS | FAIL; only PASS accepts.
"""
import ast
import contextlib
import hashlib
import io
import json
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
SCRATCH = os.path.join(ROOT, "data", "small_repairs_part3", "suite_tmp")
os.makedirs(SCRATCH, exist_ok=True)
BASE = "7b9184a95cd125c14685008cd306fe0fd9ff3040"
PROBE = os.path.join(HERE, "fixtures", "l5_fresh_process_probe.py")
I_MOD = "window_optimizer_integration_final"
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


def run_gate(name, fn):
    try:
        RESULTS.append((name, True, fn() or ""))
    except BaseException as e:   # noqa: BLE001
        RESULTS.append((name, False, f"{type(e).__name__}: {e}"[:1200]))


# ===========================================================================
# G-L5
# ===========================================================================
def _probe(*args):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([sys.executable, PROBE, *args], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=600)
    return p


def l5_run(mutant=False):
    with tmpdir("l5_") as d:
        p = _probe("open", d)
        ok_(p.returncode == 0, f"open failed: {p.stderr[-800:]}")
        ledger = json.loads(p.stdout.strip().splitlines()[-1])["ledger"]
        args = ["run", ledger] + (["--mutant"] if mutant else [])
        q = _probe(*args)
        line = [x for x in q.stdout.splitlines() if x.startswith("L5PROBE ")]
        ok_(q.returncode == 0 and line, f"run failed: rc={q.returncode} {q.stderr[-800:]}")
        return json.loads(line[-1][len("L5PROBE "):])


def g_l5():
    r = l5_run()
    ok_(r["wo_imported_before"] is False and r["wo_imported_after_watcher_import"] is False,
        f"not a fresh process: window_optimizer already imported {r}")
    ok_(r["argv_built"], f"run_step built no command: {r.get('error')}")
    ok_(r["wo_imported_after_run_step"] is True,
        "run_step did not import window_optimizer — the L-5 path was not exercised")
    ok_(r["exec_cmd_records"] == 1, f"EXEC CMD records: {r['exec_cmd_records']}")
    ok_(r["pin_records"] == 1, f"[S172-PIN] step1_pin_source records: {r['pin_records']}")
    ok_(r["root_level_after"] == r["root_level_before"],
        f"root level {r['root_level_before']} -> {r['root_level_after']}")
    return (f"fresh process; window_optimizer imported inside run_step; EXEC CMD 1, "
            f"[S172-PIN] 1, root level {r['root_level_before']} -> {r['root_level_after']}")


def g_l5_mutant_reds():
    """The base `step1_presented_configuration` (L-5 unrepaired) in the same probe."""
    r = l5_run(mutant=True)
    ok_(r["argv_built"], f"mutant run built no command: {r.get('error')}")
    return r


# ===========================================================================
# G-EVID
# ===========================================================================
RUN = "evid_t1_0000"


def _make_ledger(d, n=4, stripe=100, start=1000):
    from miner.range_miner_coordinator import MinerLedger, ST_DONE
    led = MinerLedger(os.path.join(d, "miner_ledger.db"))
    led.create_trial(RUN, 1, now=1.0)
    for ph in (1, 2):
        for i in range(n):
            sid = f"{RUN}__st{ph - 1}_s{i}"
            led.add_stripe(RUN, sid, start + i * stripe, stripe, "java_lcg", ph, now=1.0)
            led.set_stripe_state(RUN, sid, ST_DONE)
    del led
    return os.path.join(d, "miner_ledger.db"), (start, start + n * stripe)


def _listing(d):
    out = {}
    for n in sorted(os.listdir(d)):
        st = os.stat(os.path.join(d, n))
        out[n] = (st.st_size, st.st_mtime_ns)
    return out


def _content_listing(d):
    """Names, sizes, mtime_ns and sha256 of every file in d. A plain `mode=ro` reader
    rewrites a live `-shm` through mmap with size and mtime unchanged, so `_listing` alone
    cannot see it (measured 2026-11-15, docs/CLAUDE_CODE_REPORT_STEP1_CREDIT_WAL.md); an
    mtime-only change is caught too (Beta X-STEP1-CREDIT-WAL R2 Q3). This detects any
    PERSISTENT difference; a write undone between the two listings is not detectable."""
    out = {}
    for n in sorted(os.listdir(d)):
        p = os.path.join(d, n)
        st = os.stat(p)
        with open(p, "rb") as f:
            out[n] = (st.st_size, st.st_mtime_ns, hashlib.sha256(f.read()).hexdigest())
    return out


def _reader():
    import importlib
    return importlib.import_module(I_MOD)._step1_miner_sweep_evidence


def g_evid_open(reader=None):
    reader = reader or _reader()
    with tmpdir("evid_") as d:
        path, (lo, hi) = _make_ledger(d)
        before = _listing(d)
        ok_(set(before) == {"miner_ledger.db"},
            f"fixture left sidecars before the read: {sorted(before)}")
        with contextlib.redirect_stdout(io.StringIO()):
            ev = reader({"run_id": RUN, "committed": True},
                        types.SimpleNamespace(staging_dir=d, miner_output_dir=None))
        after = _listing(d)
        ok_(ev is not None and ev["swept_intervals"] == {"1": [lo, hi], "2": [lo, hi]},
            f"evidence {ev}")
        ok_(after == before, f"the read changed the directory: before {before} after {after}")
    return f"interval {{1,2: [{lo}, {hi})}} read; directory unchanged ({sorted(after)})"


def g_evid_wal(reader=None):
    reader = reader or _reader()
    with tmpdir("evidwal_") as d:
        path, (lo, hi) = _make_ledger(d)
        w = sqlite3.connect(path)
        try:
            w.execute("PRAGMA journal_mode=WAL")
            w.execute("PRAGMA wal_autocheckpoint=0")
            w.execute("UPDATE stripes SET state='staging' WHERE run_id=? AND phase=1 "
                      "AND stripe_id LIKE ?", (RUN, "%_s0"))
            w.commit()
            ok_(os.path.getsize(path + "-wal") > 0, "fixture: -wal is empty")
            # The hazard the rule exists for: an immutable read ignores the WAL and
            # would credit phase 1 although the writer's latest state says it is not done.
            im = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
            stale = im.execute("SELECT state FROM stripes WHERE stripe_id LIKE ? AND "
                               "phase=1", ("%_s0",)).fetchone()[0]
            im.close()
            ok_(stale == "done", f"fixture: the immutable view is not stale ({stale})")
            before = _content_listing(d)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                ev = reader({"run_id": RUN, "committed": True},
                            types.SimpleNamespace(staging_dir=d, miner_output_dir=None))
            after = _content_listing(d)
        finally:
            w.close()
        ok_(ev is None or "1" not in ev.get("swept_intervals", {}),
            f"a non-empty -wal was read through: evidence {ev}")
        # [Owner ruling 2026-11-15, docs/CLAUDE_CODE_REPORT_STEP1_CREDIT_WAL.md §0] The
        # blanket "non-empty -wal -> no evidence" requirement is reworded: the reader may
        # read the COMMITTED state (from a stable copy, never opening the original), so
        # phase 2 is credited and phase 1 is not; and the read changes nothing beside the
        # ledger. The fixture and the phase-1 safety assertion above are unchanged.
        ok_(ev is not None and ev.get("swept_intervals") == {"2": [lo, hi]},
            f"committed state not read as phase 2 only: evidence {ev} {buf.getvalue()!r}")
        ok_(after == before, f"the read changed the ledger directory: before {before} "
                             f"after {after}")
    return ("non-empty -wal -> phase 1 NOT credited (stale immutable view shown), phase 2 "
            "read from committed state; directory unchanged")


def _base_reader():
    src = subprocess.check_output(["git", "-C", ROOT, "show",
                                   f"{BASE}:window_optimizer_integration_final.py"]).decode()
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)
                and n.name == "_step1_miner_sweep_evidence")
    lines = src.splitlines(keepends=True)[node.lineno - 1:node.end_lineno]
    import importlib
    mod = importlib.import_module(I_MOD)
    ns = {}
    exec(compile(textwrap.dedent("".join(lines)), "<base _step1_miner_sweep_evidence>",
                 "exec"), mod.__dict__, ns)
    return ns["_step1_miner_sweep_evidence"]


# ===========================================================================
GATES = [("G-L5", g_l5), ("G-EVID-OPEN", g_evid_open), ("G-EVID-WAL", g_evid_wal)]


def _mutant(label, fn):
    try:
        det = fn()
        MUTANTS.append((label, False, f"gate stayed GREEN under the mutant: {det}"))
    except BaseException as e:   # noqa: BLE001
        MUTANTS.append((label, True, f"RED: {type(e).__name__}: {str(e)[:300]}"))


def _l5_mutant_gate():
    r = g_l5_mutant_reds()
    ok_(r["exec_cmd_records"] == 1 and r["pin_records"] == 1,
        f"EXEC CMD {r['exec_cmd_records']}, [S172-PIN] {r['pin_records']}, root "
        f"{r['root_level_before']} -> {r['root_level_after']}")
    return r


def main():
    print("=" * 78)
    print("S172 PART 3 — L-5 WATCHER logging; S7 evidence reader ledger open")
    print("=" * 78)
    for name, fn in GATES:
        run_gate(name, fn)
        n, ok, det = RESULTS[-1]
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}: {det}", flush=True)
    print("MUTANTS (each must turn its gate RED)")
    _mutant("ML5 base step1_presented_configuration (L-5 unrepaired)", _l5_mutant_gate)
    _mutant("ME1 base evidence reader (mode=ro) on a sidecar-free ledger",
            lambda: g_evid_open(_base_reader()))
    _mutant("ME2 base evidence reader with a non-empty -wal",
            lambda: g_evid_wal(_base_reader()))
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
