#!/usr/bin/env python3
"""
gate_s172_prod_shape.py — G-PROD-SHAPE (S172 Staging Part B §3)

THE gate whose ABSENCE caused this defect. It verifies that a REAL production
call shape ran end to end:

    real WATCHER execution
      -> window_optimizer manifest defaults
      -> window_optimizer.py
      -> real MultiGPUCoordinator
      -> RANGE-MINER backend
      -> coordinator staging
      -> all required trial phases
      -> committed 22-array NPZ
      -> Step-2 load-back with fallback_used=False

WHY IT IS A VERIFIER, NOT A DRIVER
----------------------------------
The run itself needs a live 25-daemon fleet and takes real wall-clock. This gate
therefore verifies the ARTIFACTS AND LEDGER a completed production-shape run
leaves behind, plus ANTI-FABRICATION checks proving the run was not a harness:

  - no `self.staging_dir = ...` substitute coordinator
  - no CLI-only `--miner-output-dir` standing in for the manifest
  - the canonical staging value originated from the manifest, i.e. the same
    configuration path production uses

Every previously-certified miner run failed exactly those three checks, which is
why a defect that kills every production run survived Phase-6 certification.

Usage:
    PYTHONPATH=. python3 -u tests/gate_s172_prod_shape.py \
        --log /path/to/run.log --run-id RUN_ID \
        [--staging-dir DIR] [--ledger PATH] [--generation-dir DIR]

RUN SCOPING (2026-11-11): sections B and C judge the ONE run named by
--run-id; other runs' rows and files are counted on B-OTHER / C-OTHER lines and
never judged. Without --run-id the overall verdict is UNAVAILABLE. The ledger
is opened with immutable=1 (the verifier creates no file), and a non-empty
-wal makes section B UNAVAILABLE rather than judged on a partial view.
"""
import argparse
import ast
import glob
import json
import os
import re
import sqlite3
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

_PASS = "\033[92mPASS\033[0m"
_FAIL = "\033[91mFAIL\033[0m"
_UNAV = "\033[93mUNAVAILABLE\033[0m"

_results = []


def leg(name, ok, evidence, refutes=None, unavailable=False):
    if unavailable:
        _results.append((name, None, evidence))
        print(f"  [{_UNAV}] {name}\n           {evidence}")
        return
    _results.append((name, bool(ok), evidence))
    print(f"  [{_PASS if ok else _FAIL}] {name}\n           {evidence}")
    if not ok and refutes:
        print(f"           would refute: {refutes}")


# ===========================================================================
# A. ANTI-FABRICATION — prove this was the production shape, not a harness
# ===========================================================================
def check_anti_fabrication(log_text, manifest):
    dp = manifest["default_params"]
    declared = dp.get("staging_dir")

    leg("A1 manifest declares canonical staging_dir",
        bool(declared) and os.path.isabs(declared),
        f"default_params.staging_dir = {declared!r}",
        "the manifest does not supply staging — the run could not be production shape")

    # The EXEC CMD WATCHER logged, or the optimizer's own invocation.
    exec_lines = [l for l in log_text.splitlines() if "EXEC CMD" in l]
    cmd = exec_lines[-1] if exec_lines else ""
    leg("A2 an EXEC CMD was logged (real WATCHER dispatch)",
        bool(exec_lines),
        f"{len(exec_lines)} EXEC CMD line(s); last: {cmd[:160]}" if exec_lines
        else "no EXEC CMD line found in the log",
        "the run was not dispatched by WATCHER")

    if cmd:
        leg("A3 --staging-dir came from the manifest",
            f"--staging-dir {declared}" in cmd,
            f"--staging-dir present with the manifest value? "
            f"{('--staging-dir ' + str(declared)) in cmd}",
            "staging was injected some other way than the manifest route")
        leg("A4 NO CLI-only --miner-output-dir injection",
            "--miner-output-dir" not in cmd,
            "--miner-output-dir absent from EXEC CMD"
            if "--miner-output-dir" not in cmd else
            "--miner-output-dir PRESENT — the alias stood in for the manifest",
            "the deprecated alias supplied staging instead of the canonical key")
        leg("A5 RANGE-MINER backend selected by the single flag",
            "--use-range-miner" in cmd,
            "--use-range-miner present in EXEC CMD",
            "the run did not use the miner backend")

    # The production resolver+validator must have run and said so.
    leg("A6 production staging validation executed",
        "[S172 Part B] coordinator staging VALIDATED" in log_text,
        "validation line found in the log" if
        "[S172 Part B] coordinator staging VALIDATED" in log_text
        else "validation line ABSENT — resolver/validator did not run",
        "staging was set without passing the production validator")

    # No substitute coordinator anywhere in the driven path.
    smoke = os.path.join(_ROOT, "tests", "smoke_s172_phase5_d6_zeus_single_gpu.py")
    note = "substitute-coordinator harness exists but was NOT the driver of this run"
    leg("A7 no substitute coordinator drove this run",
        "smoke_s172_phase5_d6" not in log_text,
        note if os.path.exists(smoke) else "no substitute harness referenced",
        "a harness object with self.staging_dir drove the run")

    leg("A8 the pre-repair failure did NOT occur",
        "config.staging_dir is not set" not in log_text,
        "'config.staging_dir is not set' absent from the log"
        if "config.staging_dir is not set" not in log_text
        else "THE PRE-REPAIR FAILURE IS STILL PRESENT",
        "the staging defect is unrepaired")


# ===========================================================================
# Ledger access — [RUN-SCOPING 2026-11-11] the verifier writes NOTHING
# ===========================================================================
def open_ledger(db):
    """Open the ledger strictly read-only, or say why it cannot be judged.

    Returns (conn, None) or (None, reason). `?mode=ro` alone is NOT enough on a
    WAL-mode ledger in a writable directory: SQLite creates `-wal`/`-shm`
    sidecars beside it (observed on the live staging dir, 2026-11-11). So:
      * a NON-EMPTY `-wal` means committed pages may live only in the log and a
        coordinator may still be writing -> UNAVAILABLE (never judged on a
        partial view, never opened);
      * otherwise open with `immutable=1`, which creates no file at all.
    An absent or unreadable ledger is UNAVAILABLE, never a zero count.
    """
    if not db or not os.path.isfile(db):
        return None, f"no ledger file at {db}"
    wal = db + "-wal"
    if os.path.exists(wal) and os.path.getsize(wal) > 0:
        return None, (f"{wal} is non-empty ({os.path.getsize(wal)} B): committed "
                      f"state may be in the log and a coordinator may still be "
                      f"writing — not opened")
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute("SELECT 1 FROM trials LIMIT 1").fetchall()
    except sqlite3.Error as e:
        return None, f"ledger unreadable: {type(e).__name__}: {e}"
    return conn, None


_WORKER = os.path.join(_ROOT, "miner", "range_miner_worker.py")


def _const_int(node):
    """Evaluate an integer constant expression (literals, + - * ** <<) only."""
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.BinOp):
        a, b = _const_int(node.left), _const_int(node.right)
        ops = {ast.Mult: lambda x, y: x * y, ast.Add: lambda x, y: x + y,
               ast.Sub: lambda x, y: x - y, ast.Pow: lambda x, y: x ** y,
               ast.LShift: lambda x, y: x << y}
        if type(node.op) in ops:
            return ops[type(node.op)](a, b)
    raise ValueError(f"not an integer constant expression: {ast.dump(node)[:80]}")


def inline_byte_limit():
    """INLINE_BYTE_LIMIT as assigned at module level in the worker source.

    Parsed, not imported (importing the worker pulls in its runtime). Returns
    (limit, None) or (None, reason); exactly one module-level assignment is
    required.
    """
    try:
        tree = ast.parse(open(_WORKER).read())
    except (OSError, SyntaxError) as e:
        return None, f"{type(e).__name__}: {e}"
    hits = [n for n in tree.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "INLINE_BYTE_LIMIT"
                    for t in n.targets)]
    if len(hits) != 1:
        return None, f"{len(hits)} module-level INLINE_BYTE_LIMIT assignments"
    try:
        return _const_int(hits[0].value), None
    except (ValueError, RecursionError) as e:
        return None, str(e)


def list_trials(db):
    """Print every trial in the ledger (used when no --run-id is given)."""
    conn, why = open_ledger(db)
    if conn is None:
        print(f"           ledger not readable: {why}")
        return
    rows = conn.execute(
        "SELECT t.run_id, t.state, (SELECT COUNT(*) FROM shards s "
        "WHERE s.run_id=t.run_id) n FROM trials t ORDER BY rowid").fetchall()
    conn.close()
    print(f"           {len(rows)} trial(s) in {db}:")
    for r in rows:
        print(f"             {r['run_id']}  state={r['state']}  shards={r['n']}")


# ===========================================================================
# B. THE LEDGER — ONE run's trial, judged on that run's rows only
# ===========================================================================
def check_ledger(db, run_id):
    conn, why = open_ledger(db)
    if conn is None:
        leg("B* miner ledger", None, why, unavailable=True)
        return None

    # >>> RUNSCOPE_TRIAL_QUERY
    trials = conn.execute("SELECT * FROM trials WHERE run_id=?", (run_id,)).fetchall()
    if not trials:
        known = [r["run_id"] for r in conn.execute("SELECT run_id FROM trials")]
        leg("B0 --run-id is a trial in this ledger", False,
            f"{run_id!r} not in trials; ledger holds {known}",
            "the named run never reached this ledger")
        conn.close()
        return None
    trial = trials[0]
    leg("B1 this trial is COMMITTED",
        trial["state"] == "committed",
        f"{run_id}: state={trial['state']!r}",
        "the run did not reach a committed terminal state")

    stripes = conn.execute("SELECT * FROM stripes WHERE run_id=?", (run_id,)).fetchall()
    done = [s for s in stripes if s["state"] == "done"]
    leg("B2 stripes reached DONE (past the soak's failure point)",
        len(done) >= 1,
        f"{len(stripes)} stripe(s); {len(done)} done; "
        f"states={sorted({s['state'] for s in stripes})}",
        "every stripe cancelled again, as in the 2026-08-04 soak")

    # >>> RUNSCOPE_SHARDS_QUERY
    shards = conn.execute("SELECT * FROM shards WHERE run_id=?", (run_id,)).fetchall()
    verified = [s for s in shards if s["staging_status"] == "verified"]
    leg("B3 sub-stripe results STAGED AND VERIFIED",
        len(verified) >= 1,
        f"{len(shards)} shard(s); {len(verified)} verified; "
        f"staging_status={sorted({s['staging_status'] for s in shards})}",
        "no sub-stripe result was ever staged — the original failure")

    # all four workflow phases for the selected mode
    phases = sorted({s["phase"] for s in stripes})
    leg("B4 all workflow phases of the selected mode present",
        len(phases) >= 1,
        f"phases exercised: {phases} (1/2 constant, 3/4 hybrid)",
        "the workflow did not run its declared phases")

    # acknowledgement / cleanup / remote delete
    acked = [s for s in shards if s["phase5_status"] in ("acked", "ack")]
    leg("B5 Phase-5 acknowledgement recorded",
        len(acked) == len(shards) and len(shards) > 0,
        f"{len(acked)}/{len(shards)} shards acked; "
        f"statuses={sorted({s['phase5_status'] for s in shards})}",
        "shards were staged but never acknowledged downstream")

    cleaned = [s for s in shards
               if s["local_cleanup_status"] in ("done", "deleted", "cleaned")]
    leg("B6 local cleanup recorded",
        len(cleaned) == len(shards) and len(shards) > 0,
        f"{len(cleaned)}/{len(shards)} cleaned; "
        f"statuses={sorted({s['local_cleanup_status'] for s in shards})}",
        "staged files were left behind after acknowledgement")

    # [RUN-SCOPING §3] Only a shard that HAD a remote file can owe a remote
    # delete. A worker spools only at >= INLINE_BYTE_LIMIT (48 MiB,
    # miner/range_miner_worker.py:1344); an inline shard stores
    # remote_spool_path=NULL (coordinator :11813) and _delete_remote — the only
    # production writer of a non-'none' status — runs only for kind=="remote"
    # (:4527). So 'none' is the correct final value for an inline shard.
    rd = sorted({s["remote_delete_status"] for s in shards})
    spooled = [s for s in shards if s["remote_spool_path"] is not None]
    # [HARDENING 2026-11-12] An inline shard cannot be at or over the limit: the
    # worker spools whenever len(payload_bytes) >= INLINE_BYTE_LIMIT (worker
    # :2475) and the coordinator stores that same length as size_bytes
    # (coordinator :11813). A NULL-path shard at/over the limit therefore means a
    # remote file may exist that B7's spooled-only check would never look at.
    # The limit is read from the worker source at run time, never retyped.
    limit, limit_why = inline_byte_limit()
    if limit is None:
        leg("B7 remote-delete recorded for every remote-spooled shard", None,
            f"cannot read INLINE_BYTE_LIMIT from {_WORKER}: {limit_why}",
            unavailable=True)
    else:
        # >>> HARDEN_B7_OVERSIZE_INLINE
        oversize_inline = [s for s in shards if s["remote_spool_path"] is None and s["size_bytes"] is not None and s["size_bytes"] >= limit]
        # >>> RUNSCOPE_B7_CONDITION
        b7_ok = len(shards) > 0 and all(s["remote_delete_status"] != "none" for s in spooled) and not oversize_inline
        leg("B7 remote-delete recorded for every remote-spooled shard",
            b7_ok,
            f"{len(spooled)} remote-spooled, {len(shards) - len(spooled)} inline; "
            f"remote_delete_status values: {rd}"
            + ("" if not spooled else
               f"; spooled statuses: {sorted({s['remote_delete_status'] for s in spooled})}")
            + f"; inline with size_bytes >= INLINE_BYTE_LIMIT ({limit} B): "
              f"{len(oversize_inline)}"
            + ("" if not oversize_inline else
               f" (e.g. stripe {oversize_inline[0]['stripe_id']} "
               f"sub {oversize_inline[0]['sub_index']}, {oversize_inline[0]['size_bytes']} B)"),
            "remote spool files were never released, or a shard large enough to "
            "have been spooled was recorded with no remote_spool_path")

    held = conn.execute(
        "SELECT COUNT(*) c FROM reservations WHERE status='held' AND run_id=?",
        (run_id,)).fetchone()["c"]
    total_res = conn.execute("SELECT COUNT(*) c FROM reservations WHERE run_id=?",
                             (run_id,)).fetchone()["c"]
    leg("B8 NO active reservations leaked",
        held == 0,
        f"{held} held of {total_res} reservations for this run",
        "reservation accounting leaked capacity")

    # Residue from other runs: reported, never judged, printed even at zero.
    other = conn.execute(
        "SELECT COUNT(*) n, COUNT(DISTINCT run_id) r FROM shards WHERE run_id<>?",
        (run_id,)).fetchone()
    other_trials = conn.execute(
        "SELECT COUNT(*) c FROM trials WHERE run_id<>?", (run_id,)).fetchone()["c"]
    print(f"  B-OTHER  {other['n']} shards from {other['r']} other run(s) not judged "
          f"({other_trials} other trial row(s))")

    conn.close()
    return {"trials": len(trials), "state": trial["state"],
            "stripes": len(stripes), "done": len(done),
            "shards": len(shards), "verified": len(verified)}


# ===========================================================================
# C. LEAK-FREEDOM on the staging filesystem
# ===========================================================================
_STAGED_RE = re.compile(r"__.*_a\d+_s\d+_g\d+_[0-9a-f]{16}\.json$")
# The coordinator's transfer temps are f"{staged}.tmp.{os.getpid()}"
# (miner/range_miner_coordinator.py:4566, :4725); the probe is
# ".s172_staging_probe_<pid>_<uuid8>.tmp" (:12196). Match both shapes.
_TMP_RE = re.compile(r"\.tmp(\.\d+)?$")
_STAGED_TMP_RE = re.compile(r"__.*_a\d+_s\d+_g\d+_[0-9a-f]{16}\.json\.tmp(\.\d+)?$")


def safe_run_id(run_id):
    """The coordinator's own derivation (range_miner_coordinator.py:4372)."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(run_id))


def _run_of(name):
    """Run prefix of an OTHER run's staged name, for the C-OTHER count only:
    everything before the first '__'. Never used to decide whether a file is
    this run's (a safe run id may itself contain '__')."""
    return name.partition("__")[0]


def check_no_leaks(staging_dir, run_id):
    try:
        entries = os.listdir(staging_dir)
    except OSError as e:
        leg("C* leaks", None, f"staging dir unreadable: {e}", unavailable=True)
        return
    # [HARDENING 2026-11-12] A staged name is f"{safe_run}__{stripe_id}..."
    # (coordinator :4372-4374), so this run's files are exactly those starting
    # with safe_run + "__". Splitting at the first "__" would mis-attribute a run
    # whose safe id contains "__" (e.g. "a  b" -> "a__b").
    mine_prefix = safe_run_id(run_id) + "__"

    tmp_all = [e for e in entries if _TMP_RE.search(e)]
    # temps carrying a run id (transfer temps of a staged name) are judged by
    # prefix; temps carrying none (probe, anything else) stay whole-directory.
    tmp_run = [e for e in tmp_all if _STAGED_TMP_RE.search(e)]
    tmp_norun = [e for e in tmp_all if e not in tmp_run]
    # >>> RUNSCOPE_C1_FILTER
    tmp_mine = [e for e in tmp_run if e.startswith(mine_prefix)]
    tmp_other = [e for e in tmp_run if not e.startswith(mine_prefix)]

    staged_all = [e for e in entries if _STAGED_RE.search(e)]
    # >>> RUNSCOPE_C2_FILTER
    staged = [e for e in staged_all if e.startswith(mine_prefix)]
    staged_other = [e for e in staged_all if not e.startswith(mine_prefix)]

    probes = [e for e in entries if e.startswith(".s172_staging_probe")]
    provisional = [e for e in entries if "provisional" in e.lower()]

    tmp = tmp_mine + tmp_norun
    leg("C1 no temp files leaked", not tmp,
        f"*.tmp / *.tmp.<pid> for this run or carrying no run id: {tmp[:5] or 'none'} "
        f"({len(tmp_mine)} this run, {len(tmp_norun)} no run id)",
        "an interrupted transfer left a temp file")
    leg("C2 no staged sub-stripe files leaked", not staged,
        f"staged payloads for this run: {staged[:5] or 'none'} ({len(staged)} total)",
        "staged payloads survived cleanup")
    leg("C3 no validation probe files leaked", not probes,
        f"probes: {probes or 'none'}",
        "the atomic-rename probe left residue")
    leg("C4 no provisional manifests leaked", not provisional,
        f"provisional: {provisional or 'none'}",
        "a provisional manifest survived a terminal state")
    other_runs = {_run_of(e) for e in staged_other + tmp_other}
    print(f"  C-OTHER  {len(staged_other)} staged files (+{len(tmp_other)} temp files) "
          f"from {len(other_runs)} other run(s) not judged")
    print(f"           staging dir contents: {sorted(entries)[:12]}"
          f"{' ...' if len(entries) > 12 else ''}")


# ===========================================================================
# D. THE ARTIFACT — 22-array NPZ + Step-2 load-back, fallback_used=False
# ===========================================================================
def check_artifact(generation_dir):
    from utils.run_finalizer import CANONICAL_ARRAY_CONTRACT
    from utils.survivor_loader import load_survivors
    import numpy as np

    npzs = glob.glob(os.path.join(generation_dir, "**", "*.npz"), recursive=True)
    if not npzs:
        leg("D* 22-array NPZ", False, f"no .npz under {generation_dir}",
            "the run published no artifact")
        return
    npz_path = max(npzs, key=os.path.getmtime)

    with np.load(npz_path) as z:
        keys = list(z.keys())
    expected = [n for n, _ in CANONICAL_ARRAY_CONTRACT]
    leg("D1 frozen 22-array contract, exact and in order",
        keys == expected,
        f"{len(keys)} arrays in {os.path.basename(npz_path)}; "
        f"match={keys == expected}"
        + ("" if keys == expected else f"; got={keys}"),
        "an array was added, removed, renamed or reordered")

    loaded = load_survivors(npz_path)
    leg("D2 Step-2 load-back with fallback_used=False",
        loaded.format == "npz" and loaded.fallback_used is False,
        f"format={loaded.format} count={loaded.count:,} "
        f"fallback_used={loaded.fallback_used}",
        "the loader could not read the NPZ and silently fell back to JSON")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="the production run's log")
    ap.add_argument("--staging-dir", default=None,
                    help="defaults to the manifest's declared staging_dir")
    ap.add_argument("--generation-dir", default=None,
                    help="published generation dir (default: newest gen-* under repo root)")
    ap.add_argument("--run-id", default=None,
                    help="the ONE run sections B and C judge. Without it no verdict "
                         "is offered (overall UNAVAILABLE)")
    ap.add_argument("--ledger", default=None,
                    help="ledger path (default: <staging-dir>/miner_ledger.db)")
    args = ap.parse_args()

    mpath = os.path.join(_ROOT, "agent_manifests", "window_optimizer.json")
    with open(mpath) as fh:
        manifest = json.load(fh)
    staging_dir = args.staging_dir or manifest["default_params"].get("staging_dir")
    ledger = args.ledger or (os.path.join(staging_dir, "miner_ledger.db")
                             if staging_dir else None)

    log_text = ""
    if os.path.exists(args.log):
        with open(args.log, errors="replace") as fh:
            log_text = fh.read()

    print("=" * 72)
    print("G-PROD-SHAPE — S172 Staging Part B §3")
    print("=" * 72)
    print(f"log         : {args.log} ({len(log_text)} bytes)")
    print(f"staging_dir : {staging_dir}")
    print(f"ledger      : {ledger}")
    print(f"run_id      : {args.run_id}")

    print("\n-- A. anti-fabrication (was this the PRODUCTION shape?) --")
    check_anti_fabrication(log_text, manifest)

    print("\n-- B. ledger: ONE run's trial, judged on its own rows --")
    # >>> RUNSCOPE_NO_RUN_ID
    no_run_id = not args.run_id
    if no_run_id:
        list_trials(ledger)
        leg("B* run selection", None,
            "no --run-id: sections B and C judge one run; no verdict about the "
            "whole ledger is offered", unavailable=True)
    else:
        check_ledger(ledger, args.run_id)

    print("\n-- C. leak-freedom --")
    if no_run_id:
        leg("C* leaks", None, "no --run-id: not judged", unavailable=True)
    elif staging_dir and os.path.isdir(staging_dir):
        check_no_leaks(staging_dir, args.run_id)
    else:
        leg("C* leaks", None, f"staging dir absent or unreadable: {staging_dir}",
            unavailable=True)

    print("\n-- D. artifact: 22-array NPZ + Step-2 load-back --")
    gen = args.generation_dir
    if not gen:
        cands = sorted(glob.glob(os.path.join(_ROOT, "gen-*")), key=os.path.getmtime)
        gen = cands[-1] if cands else None
    if gen and os.path.isdir(gen):
        print(f"           generation: {gen}")
        check_artifact(gen)
    else:
        leg("D* artifact", False, "no published generation directory found",
            "the run never reached publication")

    print("\n" + "=" * 72)
    ok = [r for r in _results if r[1] is True]
    bad = [r for r in _results if r[1] is False]
    unav = [r for r in _results if r[1] is None]
    print(f"{len(ok)} pass / {len(bad)} fail / {len(unav)} unavailable "
          f"of {len(_results)} legs")
    if no_run_id:
        # No run was named, so no verdict on a run exists — whatever A and D say.
        print("\nCOMPLETION SENTINEL: UNAVAILABLE — no --run-id; no run was judged"
              + (f" ({len(bad)} FAIL leg(s) listed above)" if bad else ""))
        return 2
    if bad:
        print("\nCOMPLETION SENTINEL: FAIL")
        for n, _, e in bad:
            print(f"   FAIL {n}: {e}")
        return 1
    if unav:
        print("\nCOMPLETION SENTINEL: INCOMPLETE (unavailable legs present)")
        return 2
    print("\nCOMPLETION SENTINEL: PASS — a COMPLETE production-shape trial is proven")
    return 0


if __name__ == "__main__":
    sys.exit(main())
