#!/usr/bin/env python3
"""G-PROD-SHAPE run scoping — gate battery (2026-11-11).

Brief: docs/CLAUDE_CODE_INSTRUCTIONS_PROD_SHAPE_RUN_SCOPING_20261111.md, as
amended by Alpha's decisions (--ledger; C1 matches {staged}.tmp.{pid} and is
judged by run prefix; non-empty -wal -> UNAVAILABLE, else immutable=1).

Every fixture is built in a temp directory: a WAL-mode SQLite ledger whose
four tables are created from the coordinator's OWN DDL, read from
miner/range_miner_coordinator.py at test time (never retyped), plus a staging
directory. The verifier is executed in-process from source text: live source,
pre-change source from BASE (git show), or a mutated copy of live source, each
compiled into a fresh namespace whose __file__ is the real verifier path. A
mutant whose target line is not found exactly once is an ERROR, never a
detection; every mutant inserts a hit sentinel proving the mutated code ran.

Nothing here opens the live ledger or the live staging directory (HYG checks
the live directory listing and ledger hash are unchanged across the suite).
Prints one line per gate and `COMPLETION SENTINEL: PASS` only if all passed.
"""

import ast
import contextlib
import hashlib
import io
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERIFIER = os.path.join(ROOT, "tests", "gate_s172_prod_shape.py")
COORD = os.path.join(ROOT, "miner", "range_miner_coordinator.py")
BASE = "b6cb789"
LIVE_STAGING = "/home/michael/miner_staging"   # observed only (listdir + sha256)

T = "distributed_config_t1_fixtureT"
X = "distributed_config_t1_fixtureX"

RESULTS = []
LAST_HITS = []   # every mutant sentinel hit across all verifier executions


class Unavailable(Exception):
    pass


class MutantNotApplied(Exception):
    pass


def record(name, verdict, detail=""):
    RESULTS.append((name, verdict, detail))
    print(f"  [{verdict:<11}] {name}  — {detail}", flush=True)


def run_gate(name, fn):
    try:
        ok, detail = fn()
        record(name, "PASS" if ok else "FAIL", detail)
    except Unavailable as e:
        record(name, "UNAVAILABLE", str(e))
    except MutantNotApplied as e:
        record(name, "ERROR", f"mutant not applied: {e}")
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc().strip().splitlines()[-1]
        record(name, "ERROR", f"{type(e).__name__}: {e} [{tb}]")


# ─────────────────────────────────────────────────────────────────────────────
# Sources
# ─────────────────────────────────────────────────────────────────────────────

def live_src():
    with open(VERIFIER) as f:
        return f.read()


_BASE_SRC = None


def base_src():
    global _BASE_SRC
    if _BASE_SRC is None:
        r = subprocess.run(["git", "show", f"{BASE}:tests/gate_s172_prod_shape.py"],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            raise Unavailable(f"git show failed: {r.stderr.strip()[:200]}")
        if "--run-id" in r.stdout:
            raise Unavailable("BASE verifier already run-scoped")
        _BASE_SRC = r.stdout
    return _BASE_SRC


def mutate(src, old_line, new_line, tag):
    """Replace the ONE line whose stripped text == old_line with a hit sentinel
    followed by new_line, both at the original indentation."""
    lines = src.splitlines(keepends=True)
    hits = [i for i, ln in enumerate(lines) if ln.strip() == old_line.strip()]
    if len(hits) != 1:
        raise MutantNotApplied(f"{tag}: target found on {len(hits)} lines: {old_line!r}")
    i = hits[0]
    ind = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
    out = (lines[:i] + [f"{ind}_MUT_HITS.append({tag!r})\n", f"{ind}{new_line}\n"]
           + lines[i + 1:])
    return "".join(out)


def run_verifier(src, argv, label, after_load=None):
    """Execute verifier source in a fresh namespace; return a result dict.

    after_load: optional {name: value} applied to the module namespace after the
    source is executed and before main() runs (fixture wiring, e.g. a missing
    worker path for B7-U). The verifier source itself is not changed."""
    ns = {"__name__": f"gate_under_test:{label}", "__file__": VERIFIER,
          "_MUT_HITS": []}
    out = io.StringIO()
    saved_argv = sys.argv
    sys.argv = ["gate_s172_prod_shape.py"] + list(argv)
    try:
        with contextlib.redirect_stdout(out):
            exec(compile(src, f"<verifier:{label}>", "exec"), ns)
            ns.update(after_load or {})
            try:
                rc = ns["main"]()
            except SystemExit as e:
                rc = e.code
    finally:
        sys.argv = saved_argv
    LAST_HITS.extend(ns["_MUT_HITS"])
    text = out.getvalue().replace("\033[92m", "").replace("\033[91m", "") \
        .replace("\033[93m", "").replace("\033[0m", "")
    legs = {}
    for name, ok, ev in ns.get("_results", []):
        legs[name.split(" ")[0]] = ("PASS" if ok is True else "FAIL" if ok is False
                                    else "UNAVAILABLE", ev)
    return {"rc": rc, "out": text, "legs": legs, "hits": ns["_MUT_HITS"],
            "sentinel": next((ln.strip() for ln in text.splitlines()
                              if "COMPLETION SENTINEL" in ln), None),
            "other": [ln.strip() for ln in text.splitlines()
                      if ln.strip().startswith(("B-OTHER", "C-OTHER"))]}


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures (coordinator DDL read at test time)
# ─────────────────────────────────────────────────────────────────────────────

def coordinator_ddl():
    tree = ast.parse(open(COORD).read())
    found = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            for t in ("stripes", "shards", "reservations", "trials"):
                if f"CREATE TABLE IF NOT EXISTS {t} (" in n.value:
                    found.setdefault(t, []).append(n.value)
    for t in ("stripes", "shards", "reservations", "trials"):
        if len(found.get(t, [])) != 1:
            raise Unavailable(f"coordinator DDL for {t}: {len(found.get(t, []))} literals")
    return {t: v[0] for t, v in found.items()}


DDL = None
TMP = tempfile.mkdtemp(prefix="prod_shape_scoping_")
EMPTY_GEN = os.path.join(TMP, "empty_generation")
os.makedirs(EMPTY_GEN)
NO_LOG = os.path.join(TMP, "no_such_run.log")
_FX = [0]


def insert(conn, table, **vals):
    cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
    row = dict(vals)
    for _cid, name, ctype, notnull, default, _pk in cols:
        if name in row or not notnull or default is not None:
            continue
        if name == "reservation_id":
            continue
        row[name] = 0 if ctype.upper() in ("INTEGER", "REAL") else ""
    names = ", ".join(row)
    q = ", ".join("?" for _ in row)
    conn.execute(f"INSERT INTO {table} ({names}) VALUES ({q})", list(row.values()))


def hexname(i):
    return hashlib.sha256(str(i).encode()).hexdigest()[:16]


def staged_name(run, i, suffix="", prefix=None):
    """prefix = the coordinator's safe_run (defaults to run, which is already
    safe for T and X); the stripe part keeps the raw run id."""
    return f"{prefix or run}__{run}__st0_s0_a0_s{i}_g0_{hexname(i)}.json{suffix}"


def build(t_state="committed", t_shards=None, x=True, t_res=("released",) * 3,
          x_res=(), extra_files=(), t_run=T):
    """Return a staging dir with miner_ledger.db (WAL mode, closed) + files.

    t_shards: list of per-shard overrides for run T (default: 3 clean inline).
    """
    global DDL
    if DDL is None:
        DDL = coordinator_ddl()
    _FX[0] += 1
    d = os.path.join(TMP, f"fx{_FX[0]:02d}")
    os.makedirs(d)
    db = os.path.join(d, "miner_ledger.db")
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA journal_mode=WAL")
    for t in ("stripes", "shards", "reservations", "trials"):
        conn.executescript(DDL[t])
    if t_shards is None:
        t_shards = [{}, {}, {}]
    insert(conn, "trials", run_id=t_run, state=t_state)
    for s in range(2):
        insert(conn, "stripes", run_id=t_run, stripe_id=f"{t_run}__st{s}", seed_start=s,
               seed_count=1, state="done", phase=1)
    for i, ov in enumerate(t_shards):
        row = dict(run_id=t_run, stripe_id=f"{t_run}__st0", attempt=0, sub_index=i,
                   worker_id="w", seed_start=i, seed_count=1, created_at=1.0,
                   staging_status="verified", phase5_status="acked",
                   local_cleanup_status="deleted", remote_delete_status="none",
                   remote_spool_path=None)
        row.update(ov)
        insert(conn, "shards", **row)
    for i, st in enumerate(t_res):
        insert(conn, "reservations", event_id=f"T{i}", run_id=t_run, stripe_id=f"{t_run}__st0",
               attempt=0, sub_index=i, staging_generation=0, size_bytes=1, status=st,
               created_at=1.0)
    if x:
        insert(conn, "trials", run_id=X, state="aborted")
        insert(conn, "stripes", run_id=X, stripe_id=f"{X}__st0", seed_start=0,
               seed_count=1, state="cancelled", phase=1)
        for i in range(2):
            insert(conn, "shards", run_id=X, stripe_id=f"{X}__st0", attempt=0,
                   sub_index=i, worker_id="w", seed_start=i, seed_count=1,
                   created_at=1.0, staging_status="verified", phase5_status="enqueued",
                   local_cleanup_status="none", remote_delete_status="none")
        for i, st in enumerate(x_res):
            insert(conn, "reservations", event_id=f"X{i}", run_id=X,
                   stripe_id=f"{X}__st0", attempt=0, sub_index=i, staging_generation=0,
                   size_bytes=1, status=st, created_at=1.0)
        for i in range(3):
            open(os.path.join(d, staged_name(X, i)), "w").write("{}")
    conn.commit()
    conn.close()
    for f in extra_files:
        open(os.path.join(d, f), "w").write("{}")
    for s in ("-wal", "-shm"):
        if os.path.exists(db + s):
            raise Unavailable(f"fixture left a {s} sidecar after close")
    return d


def argv(d, run_id=T, extra=(), log=None, gen=None):
    a = ["--log", log or NO_LOG, "--staging-dir", d, "--generation-dir", gen or EMPTY_GEN]
    if run_id is not None:
        a += ["--run-id", run_id]
    return a + list(extra)


def legs_are(res, names, verdict):
    bad = {n: res["legs"].get(n, ("MISSING",))[0] for n in names
           if res["legs"].get(n, ("MISSING",))[0] != verdict}
    return not bad, bad


BC = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "C1", "C2", "C3", "C4"]


# ─────────────────────────────────────────────────────────────────────────────
# Gates
# ─────────────────────────────────────────────────────────────────────────────

def s_red():
    r = run_verifier(base_src(), argv(build(), run_id=None), "S-RED")
    ok, _ = legs_are(r, ["B5", "B6", "C2"], "FAIL")
    return ok, (f"pre-change verifier on T+X fixture: B5={r['legs'].get('B5', ('?',))[0]} "
                f"B6={r['legs'].get('B6', ('?',))[0]} C2={r['legs'].get('C2', ('?',))[0]} "
                f"(fault reproduced: X's rows and files counted against the ledger)")


def s_a_check(r):
    ok, bad = legs_are(r, BC, "PASS")
    want = ["B-OTHER  2 shards from 1 other run(s) not judged (1 other trial row(s))",
            "C-OTHER  3 staged files (+0 temp files) from 1 other run(s) not judged"]
    return ok and r["other"] == want, bad


def s_a():
    r = run_verifier(live_src(), argv(build()), "S-A")
    ok, bad = s_a_check(r)
    return ok, f"B1-B8, C1-C4 all PASS={not bad} {bad or ''}; other-lines={r['other']}"


def _one(fixture_kwargs, leg_name, want, label, run_id=T):
    r = run_verifier(live_src(), argv(build(**fixture_kwargs), run_id=run_id), label)
    got = r["legs"].get(leg_name, ("MISSING", ""))
    return got[0] == want, f"{leg_name}={got[0]}: {got[1]}"


def s_b():
    return _one({"extra_files": [staged_name(T, 7)]}, "C2", "FAIL", "S-B")


def s_c():
    return _one({"t_shards": [{}, {"phase5_status": "enqueued"}, {}]}, "B5", "FAIL", "S-C")


def s_d():
    return _one({"t_state": "aborted"}, "B1", "FAIL", "S-D")


# ── otherwise-passing fixtures (X-0001 R1, Beta Q8) ──────────────────────────
# A log that passes A1–A8 and a generation dir whose NPZ passes D1–D2, so a gate
# can make ONE leg the only failure and require that leg alone to decide the
# overall verdict. Both are built from committed sources at test time: the
# manifest's staging_dir and the frozen CANONICAL_ARRAY_CONTRACT.
_PASSING = {}


def passing_inputs():
    if not _PASSING:
        import json
        import numpy as np
        if ROOT not in sys.path:
            sys.path.insert(0, ROOT)
        from utils.run_finalizer import CANONICAL_ARRAY_CONTRACT
        with open(os.path.join(ROOT, "agent_manifests", "window_optimizer.json")) as fh:
            declared = json.load(fh)["default_params"]["staging_dir"]
        log = os.path.join(TMP, "passing_run.log")
        with open(log, "w") as fh:
            fh.write(f"EXEC CMD: python3 window_optimizer.py --use-range-miner "
                     f"--staging-dir {declared}\n"
                     "[S172 Part B] coordinator staging VALIDATED\n")
        gen = os.path.join(TMP, "passing_generation")
        os.makedirs(gen)
        np.savez(os.path.join(gen, "survivors.npz"),
                 **{n: np.zeros(2, dtype=dt) for n, dt in CANONICAL_ARRAY_CONTRACT})
        _PASSING.update(log=log, gen=gen)
    return _PASSING["log"], _PASSING["gen"]


def fail_legs(r):
    return sorted(n for n, (v, _) in r["legs"].items() if v == "FAIL")


def unav_legs(r):
    return sorted(n for n, (v, _) in r["legs"].items() if v == "UNAVAILABLE")


def s_pass_check(src, label):
    log, gen = passing_inputs()
    r = run_verifier(src, argv(build(), log=log, gen=gen), label)
    ok = (r["rc"] == 0 and (r["sentinel"] or "").startswith("COMPLETION SENTINEL: PASS")
          and not fail_legs(r) and not unav_legs(r))
    return ok, r


def s_pass():
    ok, r = s_pass_check(live_src(), "S-PASS")
    return ok, (f"complete fixture (A-passing log, D-passing NPZ, clean run T): rc={r['rc']}; "
                f"{r['sentinel']}; {len(r['legs'])} legs, fail={fail_legs(r)} unavailable={unav_legs(r)}")


def s_e_check(src, label):
    """Unknown run id on an otherwise-passing fixture: B0 must be the ONLY failure
    and must alone make the overall verdict FAIL/exit 1. The same fixture with the
    real run id is the control and must PASS."""
    ctl_ok, ctl = s_pass_check(src, f"{label}:control")
    log, gen = passing_inputs()
    r = run_verifier(src, argv(build(), run_id="distributed_config_t1_nope", log=log, gen=gen), label)
    b0 = r["legs"].get("B0", ("MISSING", ""))
    ok = (ctl_ok and b0[0] == "FAIL" and fail_legs(r) == ["B0"] and r["rc"] == 1
          and (r["sentinel"] or "").startswith("COMPLETION SENTINEL: FAIL"))
    return ok, (f"control(run T) rc={ctl['rc']}; unknown run: B0={b0[0]} ({b0[1][:60]}…); "
                f"fail legs={fail_legs(r)}; rc={r['rc']}; {r['sentinel']}")


def s_e():
    return s_e_check(live_src(), "S-E")


def s_f_check(r):
    return (r["rc"] == 2 and r["sentinel"] is not None
            and r["sentinel"].startswith("COMPLETION SENTINEL: UNAVAILABLE")
            and T in r["out"] and X in r["out"])


def s_f():
    r = run_verifier(live_src(), argv(build(), run_id=None), "S-F")
    return s_f_check(r), f"rc={r['rc']}; {r['sentinel']}; trials listed={T in r['out'] and X in r['out']}"


def s_g():
    return _one({"t_res": ("released", "held")}, "B8", "FAIL", "S-G")


def s_h():
    return _one({"t_res": (), "x_res": ("held",)}, "B8", "PASS", "S-H")


def b7_a_check(r):
    b7 = r["legs"].get("B7", ("MISSING", ""))
    return b7[0] == "PASS" and "0 remote-spooled, 3 inline" in b7[1], b7


def b7_a():
    r = run_verifier(live_src(), argv(build()), "B7-A")
    ok, b7 = b7_a_check(r)
    return ok, f"B7={b7[0]}: {b7[1]}"


def b7_b():
    return _one({"t_shards": [{}, {"remote_spool_path": "/spool/x.json"}, {}]},
                "B7", "FAIL", "B7-B")


def b7_c():
    ok, d = _one({"t_shards": [{}, {"remote_spool_path": "/spool/x.json",
                                    "remote_delete_status": "deleted"}, {}]},
                 "B7", "PASS", "B7-C")
    return ok and "1 remote-spooled, 2 inline" in d, d


def b7_d():
    return _one({"t_shards": [], "t_res": ()}, "B7", "FAIL", "B7-D")


TMP_T = staged_name(T, 9, ".tmp.12345")
TMP_X = staged_name(X, 9, ".tmp.54321")


def c1_tmp():
    r = run_verifier(live_src(), argv(build(extra_files=[TMP_T])), "C1-TMP")
    c1 = r["legs"].get("C1", ("MISSING", ""))
    r2 = run_verifier(live_src(), argv(build(extra_files=[TMP_X])), "C1-TMP-other")
    c1x = r2["legs"].get("C1", ("MISSING", ""))
    ok = (c1[0] == "FAIL" and c1x[0] == "PASS"
          and "C-OTHER  3 staged files (+1 temp files) from 1 other run(s) not judged" in r2["other"])
    return ok, (f"T's .tmp.<pid> -> C1={c1[0]} ({c1[1]}); X's .tmp.<pid> -> C1={c1x[0]}, "
                f"{[o for o in r2['other'] if o.startswith('C-')]}")


def c1_red():
    r = run_verifier(base_src(), argv(build(extra_files=[TMP_T]), run_id=None), "C1-RED")
    c1 = r["legs"].get("C1", ("MISSING", ""))
    return c1[0] == "PASS", (f"pre-change verifier with {TMP_T[-24:]} present: C1={c1[0]} "
                             f"(missed: endswith('.tmp') cannot match .tmp.<pid>)")


def wal_a_check(src, label):
    d = build()
    db = os.path.join(d, "miner_ledger.db")
    w = sqlite3.connect(db)
    w.execute("PRAGMA wal_autocheckpoint=0")
    insert(w, "trials", run_id="distributed_config_t1_inflight", state="running")
    w.commit()
    try:
        size = os.path.getsize(db + "-wal")
        if size == 0:
            raise Unavailable("could not produce a non-empty -wal")
        r = run_verifier(src, argv(d), label)
    finally:
        w.close()
    bstar = r["legs"].get("B*", ("MISSING", ""))
    judged = [n for n in ("B1", "B5", "B6", "B7", "B8") if n in r["legs"]]
    ok = bstar[0] == "UNAVAILABLE" and not judged and r["rc"] != 0
    return ok, f"-wal={size} B; B*={bstar[0]} ({bstar[1][:70]}…); judged={judged}; rc={r['rc']}"


def wal_a():
    return wal_a_check(live_src(), "WAL-A")


def wal_b_check(src, label):
    out = []
    ok_all = True
    for variant in ("no -wal", "empty -wal"):
        d = build()
        db = os.path.join(d, "miner_ledger.db")
        if variant == "empty -wal":
            open(db + "-wal", "wb").close()
        before = {e: os.path.getsize(os.path.join(d, e)) for e in os.listdir(d)}
        sha_before = hashlib.sha256(open(db, "rb").read()).hexdigest()
        r = run_verifier(src, argv(d), f"{label}:{variant}")
        after = {e: os.path.getsize(os.path.join(d, e)) for e in os.listdir(d)}
        sha_after = hashlib.sha256(open(db, "rb").read()).hexdigest()
        ok, _ = s_a_check(r)
        unchanged = before == after and sha_before == sha_after
        ok_all &= ok and unchanged
        out.append(f"{variant}: legs judged+PASS={ok}, dir unchanged={unchanged}"
                   + ("" if unchanged else f" (new: {sorted(set(after) - set(before))})"))
    return ok_all, "; ".join(out)


def wal_b():
    return wal_b_check(live_src(), "WAL-B")


# ── hardening 2026-11-12: run ids whose safe form contains "__"; oversize inline ─

SEP_RUN = "a  b"          # coordinator safe form is "a__b" (two unsafe spaces)
SEP_PREFIX = "a__b"
SEP_STAGED = staged_name(SEP_RUN, 7, prefix=SEP_PREFIX)
SEP_TMP = staged_name(SEP_RUN, 8, ".tmp.777", prefix=SEP_PREFIX)


def c_sep_check(src, label, files):
    r = run_verifier(src, argv(build(t_run=SEP_RUN, extra_files=files), run_id=SEP_RUN), label)
    return r, r["legs"].get("C1", ("MISSING", ""))[0], r["legs"].get("C2", ("MISSING", ""))[0]


def c_sep():
    r, c1, c2 = c_sep_check(live_src(), "C-SEP", [SEP_STAGED, SEP_TMP])
    _, c1_ctl, c2_ctl = c_sep_check(live_src(), "C-SEP-control", [])
    ok = c2 == "FAIL" and c1 == "FAIL" and c2_ctl == "PASS" and c1_ctl == "PASS"
    return ok, (f"run {SEP_RUN!r} (safe {SEP_PREFIX!r}) with 1 leaked staged + 1 leaked "
                f"temp: C2={c2} C1={c1}; control with neither: C2={c2_ctl} C1={c1_ctl}; "
                f"C2 evidence: {r['legs'].get('C2', ('', ''))[1][:90]}")


def c_sep_red():
    # the pre-hardening attribution: compare the text before the first "__"
    src = m("staged = [e for e in staged_all if e.startswith(mine_prefix)]",
            "staged = [e for e in staged_all if _run_of(e) == safe_run_id(run_id)]",
            "RED-C2")
    src = mutate(src, "tmp_mine = [e for e in tmp_run if e.startswith(mine_prefix)]",
                 "tmp_mine = [e for e in tmp_run if _run_of(e) == safe_run_id(run_id)]",
                 "RED-C1")
    r, c1, c2 = c_sep_check(src, "C-SEP-RED", [SEP_STAGED, SEP_TMP])
    return (c2 == "PASS" and c1 == "PASS" and sorted(r["hits"]) == ["RED-C1", "RED-C2"],
            f"partition-at-first-'__' attribution, hits={r['hits']}: C2={c2} C1={c1} "
            f"(missed: the leaked files are counted as run 'a')")


def worker_inline_limit():
    """INLINE_BYTE_LIMIT from the worker source (independent of the verifier)."""
    w = os.path.join(ROOT, "miner", "range_miner_worker.py")
    hits = [n for n in ast.parse(open(w).read()).body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "INLINE_BYTE_LIMIT" for t in n.targets)]
    if len(hits) != 1:
        raise Unavailable(f"{len(hits)} INLINE_BYTE_LIMIT assignments in the worker")
    return eval(compile(ast.Expression(hits[0].value), w, "eval"), {"__builtins__": {}})


def b7_e_arms(src, label):
    L = worker_inline_limit()
    arms = [("inline at limit", {"size_bytes": L}, "FAIL"),
            ("inline at limit-1", {"size_bytes": L - 1}, "PASS"),
            ("spooled at limit, deleted", {"size_bytes": L, "remote_spool_path": "/spool/x.json",
                                           "remote_delete_status": "deleted"}, "PASS")]
    out, ok = [], True
    for name, ov, want in arms:
        r = run_verifier(src, argv(build(t_shards=[{}, ov, {}])), f"{label}:{name}")
        got = r["legs"].get("B7", ("MISSING", ""))
        ok &= got[0] == want
        out.append(f"{name} -> B7={got[0]} (want {want})")
    return ok, L, out


def b7_e():
    ok, L, out = b7_e_arms(live_src(), "B7-E")
    return ok, f"INLINE_BYTE_LIMIT={L} B (read from worker); " + "; ".join(out)


def b7_u_check(src, label):
    """X-0001 R1, Beta Q4: when INLINE_BYTE_LIMIT cannot be read, B7 is UNAVAILABLE
    and the run cannot pass. Arms: worker file absent; two module-level assignments.
    Uses the otherwise-passing fixture, so B7 alone decides the overall verdict."""
    log, gen = passing_inputs()
    ambiguous = os.path.join(TMP, "worker_two_limits.py")
    with open(ambiguous, "w") as fh:
        fh.write("INLINE_BYTE_LIMIT = 1\nINLINE_BYTE_LIMIT = 2\n")
    arms = [("worker absent", os.path.join(TMP, "no_such_worker.py")),
            ("two assignments", ambiguous)]
    out, ok = [], True
    for name, path in arms:
        r = run_verifier(src, argv(build(), log=log, gen=gen), f"{label}:{name}",
                         after_load={"_WORKER": path})
        b7 = r["legs"].get("B7", ("MISSING", ""))
        arm_ok = (b7[0] == "UNAVAILABLE" and "cannot read INLINE_BYTE_LIMIT" in b7[1]
                  and unav_legs(r) == ["B7"] and not fail_legs(r) and r["rc"] == 2)
        ok &= arm_ok
        out.append(f"{name} -> B7={b7[0]} rc={r['rc']} unavailable={unav_legs(r)} fail={fail_legs(r)}")
    return ok, out


def b7_u():
    ok, out = b7_u_check(live_src(), "B7-U")
    return ok, "; ".join(out)


# ── mutants (copies of LIVE source) ──────────────────────────────────────────

def m(old, new, tag):
    return mutate(live_src(), old, new, tag)


def mut_1():
    src = m('shards = conn.execute("SELECT * FROM shards WHERE run_id=?", (run_id,)).fetchall()',
            'shards = conn.execute("SELECT * FROM shards").fetchall()', "MUT-1")
    r = run_verifier(src, argv(build()), "MUT-1")
    red = not s_a_check(r)[0]
    return red and r["hits"] == ["MUT-1"], (
        f"shards query unscoped, hits={r['hits']}; S-A {'RED' if red else 'still green'}: "
        f"B5={r['legs'].get('B5', ('?',))[0]} B6={r['legs'].get('B6', ('?',))[0]}")


def mut_2():
    src = m("staged = [e for e in staged_all if e.startswith(mine_prefix)]",
            "staged = list(staged_all)", "MUT-2")
    r = run_verifier(src, argv(build()), "MUT-2")
    red = not s_a_check(r)[0]
    return red and r["hits"] == ["MUT-2"], (
        f"C2 filter removed, hits={r['hits']}; S-A {'RED' if red else 'still green'}: "
        f"C2={r['legs'].get('C2', ('?',))[0]}")


def mut_3():
    src = m('b7_ok = len(shards) > 0 and all(s["remote_delete_status"] != "none" for s in spooled) and not oversize_inline',
            'b7_ok = all(s["remote_delete_status"] != "none" for s in shards) and len(shards) > 0',
            "MUT-3")
    r = run_verifier(src, argv(build()), "MUT-3")
    red = not b7_a_check(r)[0]
    return red and r["hits"] == ["MUT-3"], (
        f"old B7 condition restored, hits={r['hits']}; B7-A {'RED' if red else 'still green'}: "
        f"B7={r['legs'].get('B7', ('?',))[0]}")


def mut_b7l():
    src = m('oversize_inline = [s for s in shards if s["remote_spool_path"] is None and s["size_bytes"] is not None and s["size_bytes"] >= limit]',
            "oversize_inline = []", "MUT-B7L")
    LAST_HITS.clear()
    ok, _, out = b7_e_arms(src, "MUT-B7L")
    hits = list(LAST_HITS)
    return (not ok and hits.count("MUT-B7L") == 3,
            f"oversize-inline condition removed, hits={hits.count('MUT-B7L')}x; "
            f"B7-E {'RED' if not ok else 'still green'}: " + "; ".join(out))


def mut_4():
    src = m("# No run was named, so no verdict on a run exists — whatever A and D say.",
            'print("\\nCOMPLETION SENTINEL: PASS"); return 0', "MUT-4")
    r = run_verifier(src, argv(build(), run_id=None), "MUT-4")
    red = not s_f_check(r)
    return red and r["hits"] == ["MUT-4"], (
        f"no-run-id path returns PASS, hits={r['hits']}; S-F {'RED' if red else 'still green'}: "
        f"rc={r['rc']} {r['sentinel']}")


def mut_5():
    src = m("if os.path.exists(wal) and os.path.getsize(wal) > 0:", "if False:", "MUT-5")
    LAST_HITS.clear()
    ok, detail = wal_a_check(src, "MUT-5")
    hits = list(LAST_HITS)
    return (not ok and hits.count("MUT-5") >= 1,
            f"non-empty -wal check removed, hits={hits}; WAL-A {'RED' if not ok else 'still green'}: {detail}")


def mut_6():
    src = m('conn = sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)',
            'conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)', "MUT-6")
    LAST_HITS.clear()
    ok, detail = wal_b_check(src, "MUT-6")
    hits = list(LAST_HITS)
    return (not ok and hits.count("MUT-6") >= 2,
            f"immutable=1 dropped (plain mode=ro), hits={hits}; WAL-B {'RED' if not ok else 'still green'}: {detail}")


def mut_c1():
    src = m(r'_TMP_RE = re.compile(r"\.tmp(\.\d+)?$")',
            r'_TMP_RE = re.compile(r"\.tmp$")', "MUT-C1")
    r = run_verifier(src, argv(build(extra_files=[TMP_T])), "MUT-C1")
    c1 = r["legs"].get("C1", ("MISSING",))[0]
    return c1 == "PASS" and r["hits"] == ["MUT-C1"], (
        f"C1 back to '*.tmp' only, hits={r['hits']}; C1-TMP {'RED' if c1 == 'PASS' else 'still green'}: C1={c1}")


def mut_se():
    # Beta's X-0001 R1 Q8 probe: the final failure collection ignores B0.
    src = m("bad = [r for r in _results if r[1] is False]",
            'bad = [r for r in _results if r[1] is False and not r[0].startswith("B0")]', "MUT-SE")
    LAST_HITS.clear()
    ok, detail = s_e_check(src, "MUT-SE")
    hits = list(LAST_HITS)
    return (not ok and hits.count("MUT-SE") == 2,
            f"B0 dropped from the final verdict, hits={hits.count('MUT-SE')}x; "
            f"S-E {'RED' if not ok else 'still green'}: {detail}")


def mut_b7u():
    src = m("if limit is None:", "if False:", "MUT-B7U")
    LAST_HITS.clear()
    try:
        ok, out = b7_u_check(src, "MUT-B7U")
        detail = "; ".join(out)
    except TypeError as e:  # limit=None reaching the >= comparison is also a detection
        ok, detail = False, f"TypeError: {e}"
    hits = list(LAST_HITS)
    return (not ok and hits.count("MUT-B7U") >= 1,
            f"B7 UNAVAILABLE branch removed, hits={hits.count('MUT-B7U')}x; "
            f"B7-U {'RED' if not ok else 'still green'}: {detail}")


# ─────────────────────────────────────────────────────────────────────────────

def live_fingerprint():
    try:
        ents = sorted(os.listdir(LIVE_STAGING))
        sizes = {e: os.path.getsize(os.path.join(LIVE_STAGING, e)) for e in ents}
        sha = hashlib.sha256(open(os.path.join(LIVE_STAGING, "miner_ledger.db"), "rb").read()).hexdigest()
        return (tuple(ents), tuple(sorted(sizes.items())), sha)
    except OSError as e:
        return ("UNREADABLE", str(e))


def main():
    before = live_fingerprint()
    print("G-PROD-SHAPE RUN SCOPING — gate battery", flush=True)
    print(f"  base={BASE}  temp={TMP}", flush=True)
    gates = [("S-RED", s_red), ("S-A", s_a), ("S-B", s_b), ("S-C", s_c), ("S-D", s_d),
             ("S-E", s_e), ("S-F", s_f), ("S-G", s_g), ("S-H", s_h),
             ("B7-A", b7_a), ("B7-B", b7_b), ("B7-C", b7_c), ("B7-D", b7_d),
             ("C1-TMP", c1_tmp), ("C1-RED", c1_red), ("WAL-A", wal_a), ("WAL-B", wal_b),
             ("C-SEP", c_sep), ("C-SEP-RED", c_sep_red), ("B7-E", b7_e),
             ("S-PASS", s_pass), ("B7-U", b7_u),
             ("MUT-1", mut_1), ("MUT-2", mut_2), ("MUT-3", mut_3), ("MUT-4", mut_4),
             ("MUT-5", mut_5), ("MUT-6", mut_6), ("MUT-C1", mut_c1),
             ("MUT-B7L", mut_b7l), ("MUT-SE", mut_se), ("MUT-B7U", mut_b7u)]
    for n, f in gates:
        run_gate(n, f)
    after = live_fingerprint()
    run_gate("HYG-LIVE-UNTOUCHED", lambda: (
        before == after and before[0] != "UNREADABLE",
        f"live staging listing+sizes+ledger sha256 unchanged={before == after}; "
        f"ledger sha256={before[2][:16] if len(before) == 3 else before}"))
    shutil.rmtree(TMP, ignore_errors=True)

    n_pass = sum(1 for _, v, _ in RESULTS if v == "PASS")
    print(f"  {n_pass}/{len(RESULTS)} PASS", flush=True)
    if n_pass == len(RESULTS):
        print("COMPLETION SENTINEL: PASS", flush=True)
        return 0
    print("COMPLETION SENTINEL: NOT REACHED — "
          + ", ".join(f"{n}={v}" for n, v, _ in RESULTS if v != "PASS"), flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
