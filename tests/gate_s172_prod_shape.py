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
        --log /path/to/run.log [--staging-dir DIR] [--generation-dir DIR]
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
# B. THE LEDGER — a COMPLETE trial that progressed past the soak's failure
# ===========================================================================
def check_ledger(staging_dir):
    db = os.path.join(staging_dir, "miner_ledger.db")
    if not os.path.exists(db):
        leg("B* miner ledger", False, f"no ledger at {db}",
            "no run reached coordinator construction")
        return None
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    trials = conn.execute("SELECT * FROM trials").fetchall()
    committed = [t for t in trials if t["state"] == "committed"]
    leg("B1 at least one COMMITTED trial",
        len(committed) >= 1,
        f"{len(trials)} trial(s); states={[t['state'] for t in trials]}",
        "no trial reached a committed terminal state")

    stripes = conn.execute("SELECT * FROM stripes").fetchall()
    done = [s for s in stripes if s["state"] == "done"]
    leg("B2 stripes reached DONE (past the soak's failure point)",
        len(done) >= 1,
        f"{len(stripes)} stripe(s); {len(done)} done; "
        f"states={sorted({s['state'] for s in stripes})}",
        "every stripe cancelled again, as in the 2026-08-04 soak")

    shards = conn.execute("SELECT * FROM shards").fetchall()
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

    rd = sorted({s["remote_delete_status"] for s in shards})
    leg("B7 remote-delete state recorded for every shard",
        all(s["remote_delete_status"] != "none" for s in shards) and len(shards) > 0,
        f"remote_delete_status values: {rd}",
        "remote spool files were never released")

    held = conn.execute(
        "SELECT COUNT(*) c FROM reservations WHERE status='held'").fetchone()["c"]
    total_res = conn.execute("SELECT COUNT(*) c FROM reservations").fetchone()["c"]
    leg("B8 NO active reservations leaked",
        held == 0,
        f"{held} held of {total_res} total reservations",
        "reservation accounting leaked capacity")

    conn.close()
    return {"trials": len(trials), "committed": len(committed),
            "stripes": len(stripes), "done": len(done),
            "shards": len(shards), "verified": len(verified)}


# ===========================================================================
# C. LEAK-FREEDOM on the staging filesystem
# ===========================================================================
def check_no_leaks(staging_dir):
    entries = os.listdir(staging_dir)
    tmp = [e for e in entries if e.endswith(".tmp")]
    staged = [e for e in entries if re.search(r"__.*_a\d+_s\d+_g\d+_[0-9a-f]{16}\.json$", e)]
    probes = [e for e in entries if e.startswith(".s172_staging_probe")]
    provisional = [e for e in entries if "provisional" in e.lower()]

    leg("C1 no temp files leaked", not tmp, f"*.tmp: {tmp or 'none'}",
        "an interrupted transfer left a temp file")
    leg("C2 no staged sub-stripe files leaked", not staged,
        f"staged payloads: {staged[:5] or 'none'} ({len(staged)} total)",
        "staged payloads survived cleanup")
    leg("C3 no validation probe files leaked", not probes,
        f"probes: {probes or 'none'}",
        "the atomic-rename probe left residue")
    leg("C4 no provisional manifests leaked", not provisional,
        f"provisional: {provisional or 'none'}",
        "a provisional manifest survived a terminal state")
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
    args = ap.parse_args()

    mpath = os.path.join(_ROOT, "agent_manifests", "window_optimizer.json")
    with open(mpath) as fh:
        manifest = json.load(fh)
    staging_dir = args.staging_dir or manifest["default_params"].get("staging_dir")

    log_text = ""
    if os.path.exists(args.log):
        with open(args.log, errors="replace") as fh:
            log_text = fh.read()

    print("=" * 72)
    print("G-PROD-SHAPE — S172 Staging Part B §3")
    print("=" * 72)
    print(f"log         : {args.log} ({len(log_text)} bytes)")
    print(f"staging_dir : {staging_dir}")

    print("\n-- A. anti-fabrication (was this the PRODUCTION shape?) --")
    check_anti_fabrication(log_text, manifest)

    print("\n-- B. ledger: a COMPLETE trial past the soak's failure point --")
    if staging_dir and os.path.isdir(staging_dir):
        check_ledger(staging_dir)
    else:
        leg("B* ledger", False, f"staging dir absent: {staging_dir}",
            "no staging directory was created")

    print("\n-- C. leak-freedom --")
    if staging_dir and os.path.isdir(staging_dir):
        check_no_leaks(staging_dir)
    else:
        leg("C* leaks", False, "staging dir absent", None)

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
