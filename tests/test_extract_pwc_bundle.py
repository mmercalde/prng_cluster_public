#!/usr/bin/env python3
"""test_extract_pwc_bundle.py — scripts/extract_pwc_bundle.py on a synthetic PWC run tree.
CPU only; no GPU, no rig, no live tree. Files go under <root>/logs/ (git-ignored).

Run from the repository root:   python3 -u tests/test_extract_pwc_bundle.py

The fixture is built with the REAL producers where they are pure: the generation by
`utils.run_finalizer.finalize_run` (provenance.json + the 22-array NPZ), the ledger with
the real DDL (`utils.seed_coverage_ledger._CREATE_TABLE_SQL`,
`utils.step1_search_ledger._DDL`). The four pass files have the shape
`PersistentWorkerCoordinator.run_sieve_pass` writes (persistent_worker_coordinator.py
:2486-2516), match rates float32 as the GPU kernel returns them (sieve_filter.py:252,318).

Gates
  G-CLEAN       two identical PWC run trees -> two bundles -> comparator exit 0 SAME;
                every section filled; rates exact float32; NPZ copied and hash-checked;
                credit from the ledger; swept intervals from the rebuilt completion record
  G-READONLY    the run tree is byte-identical after extraction; no -wal/-shm sidecar
  G-SUMMARY     a phase over 100K survivors (summary file only) -> that phase null; vs a
                full bundle or another summary bundle: exit 3, that phase NOT COMPARED
  O-*           [Stage 4 run, 2026-11-21] the pass files' _t<N> is the record ordinal, not the
                Optuna trial: discovered and bound by evidence (link 1/0/0/0, NPZ trial_number,
                optional run log); ambiguous, split, mismatched or asserted-wrong ordinals refused
  R1-R13        refusals: parent generation, NPZ hash, missing pass, not TCP, interval,
                incomplete pass, float64 rate, NPZ != union of sets, not a PWC run, live
                path, stale pass file, non-empty out dir, no ledger
"""
import contextlib
import copy
import io
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
OUT = os.path.join(ROOT, "logs", "test_extract_pwc_bundle")
TOOL = os.path.join(ROOT, "scripts", "extract_pwc_bundle.py")
CMP = os.path.join(ROOT, "scripts", "compare_engine_outputs.py")

import importlib.util                                                        # noqa: E402
_spec = importlib.util.spec_from_file_location("extract_pwc_bundle", TOOL)
X = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(X)
CEO = X.CEO
from pathlib import Path                                                     # noqa: E402
from utils.run_finalizer import finalize_run                                # noqa: E402
from utils.seed_coverage_ledger import _CREATE_TABLE_SQL as COV_DDL         # noqa: E402
from utils.seed_coverage_ledger import PUBLICATION_STATUS_CERTIFIED as CERTIFIED  # noqa: E402
from utils.step1_search_ledger import _DDL as S1_DDL, T_LINK                # noqa: E402
from persistent_worker_coordinator import (                                 # noqa: E402
    pwc_threshold_record, pwc_hybrid_threshold_record)

S = 2147483648                  # 2^31, Stage 4's interval start
N = 1000
WS, OFF = 12, 25
# A fresh run's first trial: Optuna trial 0, record ordinal 1 (window_optimizer_integration_final
# .py:3273, :3296, :3316). PWC names its pass files _t1 and the NPZ rows carry trial_number 1.
ORD = 1
F32 = lambda x: float(np.float32(x))                                        # noqa: E731

# seed offsets -> match rate, per phase (constant: 1 fwd, 2 rev; variable: 3 fwd, 4 rev)
PHASE_DATA = {
    "1": [(5, 0.75), (1, 0.8333333), (2, 0.9166667), (0, 0.75)],
    "2": [(2, 0.8333333), (1, 0.75), (7, 0.75)],
    "3": [(2, 0.75), (10, 0.8333333)],
    "4": [(10, 0.9166667), (2, 0.75), (11, 0.75)],
}
PASS_TYPE = {"1": "java_lcg", "2": "java_lcg_reverse", "3": "java_lcg_hybrid",
             "4": "java_lcg_hybrid_reverse"}
PARAMS = {"prng_type": "java_lcg", "seed_start": 0, "max_seeds": N, "test_both_modes": True,
          "use_range_miner": False, "use_persistent_workers": True, "pwc_transport": "tcp",
          "pwc_coordinator_address": "192.168.3.177", "seed_cap_amd": 25000,
          "worker_pool_size": 8, "window_trials": 1, "n_parallel": 1,
          "warm_start_window": WS, "warm_start_offset": OFF, "warm_start_session_idx": 1,
          "warm_start_fwd_thresh": 0.71, "warm_start_rev_thresh": 0.47,
          "warm_start_skip_min": 0, "warm_start_skip_max": 16}
DATASET_SHA = "5136" * 16


def cand(seed, mode, fwd, rev):
    sc = F32((fwd + rev) / 2)
    return {"seed": seed, "forward_match_rate": fwd, "reverse_match_rate": rev, "score": sc,
            "window_size": WS, "offset": OFF, "skip_min": 0, "skip_max": 16, "skip_range": 16,
            "sessions": ["evening"], "trial_number": ORD, "prng_base": "java_lcg",
            "skip_mode": mode, "prng_type": "java_lcg" if mode == "constant" else "java_lcg_hybrid",
            "forward_count": 4.0, "reverse_count": 3.0, "bidirectional_count": 2.0,
            "intersection_count": 2.0, "intersection_ratio": 0.4, "forward_only_count": 2.0,
            "reverse_only_count": 1.0, "survivor_overlap_ratio": 0.5,
            "bidirectional_selectivity": 1.3333, "intersection_weight": 0.4}


def accounting(ph, **over):
    a = {"schema": "pwc_pass_accounting_v1", "prng_type": PASS_TYPE[ph],
         "requested_interval": [S, S + N], "chunk_size": 250, "workers_per_rig": 8,
         "planned": 4, "dispatched": 4, "completed": 4, "failed": 0, "timed_out": 0,
         "not_dispatched": 0, "missing": 0, "seed_range_unreported": 0,
         "pass_progress_timeout": False, "completed_intervals": [[S, S + N]],
         "incomplete_chunks": [], "incomplete_chunks_truncated": False, "complete": True,
         "transport": "tcp", "threshold": threshold_rec(ph)}
    a.update(over)
    return a


def threshold_rec(ph, **over):
    """The producer's own record (persistent_worker_coordinator.py:579-641), as a clean
    TCP pass leaves it: every chunk reported the trial's threshold."""
    req = PARAMS["warm_start_fwd_thresh"] if ph in ("1", "3") else PARAMS["warm_start_rev_thresh"]
    if ph in ("1", "2"):
        r = pwc_threshold_record(PASS_TYPE[ph], req, [req], [req], 0)
    else:
        r = pwc_hybrid_threshold_record(PASS_TYPE[ph], req, [req], [req], [round(req, 4)], 0)
    r.update(over)
    return r


def pass_doc(ph, data, summary=False, **acc_over):
    seeds = [S + o for o, _ in data]
    rates = [F32(r) for _, r in data]
    d = {"survivors": seeds, "match_rates": rates, "skip_sequences": [], "strategy_ids": [],
         "total_tested": N, "survivor_count": len(seeds), "prng_type": PASS_TYPE[ph],
         "threshold": 0.71, "failed_chunks": 0, "total_chunks": 4, "seed_start": S,
         "seed_end": S + N, "pass_complete": True, "pwc_pass_accounting": accounting(ph, **acc_over)}
    if summary:
        d = {k: d[k] for k in ("survivor_count", "total_tested", "prng_type", "threshold",
                                "failed_chunks", "total_chunks", "pass_complete",
                                "pwc_pass_accounting")}
        d["note"] = "Full survivors omitted (count > 100K) — see NPZ accumulator"
    return d


def candidates(phase_data):
    rates = {ph: {S + o: F32(r) for o, r in v} for ph, v in phase_data.items()}
    out = []
    for mode, (f, r) in (("constant", ("1", "2")), ("variable", ("3", "4"))):
        for s in sorted(set(rates[f]) & set(rates[r])):
            out.append(cand(s, mode, rates[f][s], rates[r][s]))
    return out


def make_tree(tag, phase_data=None, pass_over=None, summary=(), cand_data=None,
              ledger=True, params=None, twice=False, ordinals=None, cand_ord=None, link_trials=(1, 0, 0, 0)):
    """A finished PWC run tree: results/ pass files, then the published generation, then
    the ledger rows (the order the run writes them)."""
    phase_data = copy.deepcopy(phase_data or PHASE_DATA)
    pass_over = pass_over or {}
    r = os.path.join(OUT, tag, "run")
    os.makedirs(os.path.join(r, "results"))
    for ph, data in phase_data.items():
        doc = pass_doc(ph, data, summary=ph in summary, **pass_over.get(ph, {}))
        for o in (ordinals or {}).get(ph, [ORD]):
            with open(X.pass_file(os.path.join(r, "results"), ph, WS, OFF, o), "w") as f:
                json.dump(doc, f)
    time.sleep(0.02)
    kw = dict(output_root=Path(r), run_id=f"step1_java_lcg_{S}", prng_base="java_lcg",
              skip_modes_executed=("constant", "variable"), seed_start=S, seed_count=N,
              repository_commit="6d75f71" + "0" * 33, repository_tree_clean=True)
    cands = candidates(cand_data or phase_data)
    if cand_ord is not None:
        for c in cands:
            c["trial_number"] = cand_ord
    art = finalize_run(cands, **kw)
    if twice:
        time.sleep(0.02)
        art = finalize_run(candidates(cand_data or phase_data), **dict(kw, run_id="again"))
    if ledger:
        con = sqlite3.connect(os.path.join(r, "prng_analysis.db"))
        con.executescript(COV_DDL)
        con.execute(S1_DDL[T_LINK])
        con.execute("INSERT INTO certified_coverage VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    ("cov" + tag, art.run_id, "study", "java_lcg", "constant,variable",
                     "v1.1-stratum", S, S + N, DATASET_SHA, "6d75f71", art.artifact_sha256,
                     art.generation_id, CERTIFIED, "2026-11-21T00:00:00Z"))
        con.execute(f"INSERT INTO {T_LINK} VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    ("link" + tag, "cov" + tag, "s1s-" + tag, "chunk", "java_lcg",
                     "constant,variable", S, S + N, art.generation_id, *link_trials,
                     "2026-11-21T00:00:01Z"))
        con.commit()
        con.close()
    pp = os.path.join(OUT, tag, "params.json")
    with open(pp, "w") as f:
        json.dump(params or PARAMS, f)
    return r, pp, art


def extract(tag, r, pp, **kw):
    out = os.path.join(OUT, tag, "bundle")
    return X.extract(r, pp, out, live_paths=kw.pop("live_paths", ()), **kw)


def tree_digest(d):
    out = {}
    for base, _, files in os.walk(d):
        for f in files:
            p = os.path.join(base, f)
            if os.path.islink(p):
                out[p] = "link:" + os.readlink(p)
            else:
                out[p] = X._sha256(p)
    return out


def cmp(pa, pb):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = CEO.main([pa, pb])
    return rc, CEO.compare(CEO.load_bundle(pa), CEO.load_bundle(pb)), buf.getvalue()


# --------------------------------------------------------------------------- gates
def g_clean():
    ra, pa, art = make_tree("clean_a")
    rb, pb, _ = make_tree("clean_b")
    ba, bja = extract("clean_a", ra, pa)
    bb, _ = extract("clean_b", rb, pb)
    rc, rep, txt = cmp(ba, bb)
    assert rc == 0 and rep["same"], (rc, rep["not_compared"], txt[-600:])
    # contents
    assert bja["phases"]["1"]["seeds"] == sorted(S + o for o, _ in PHASE_DATA["1"])
    for ph, data in PHASE_DATA.items():
        want = dict((S + o, F32(r)) for o, r in data)
        got = dict(zip(bja["phases"][ph]["seeds"], bja["phases"][ph]["match_rates"]))
        assert got == want, (ph, got, want)
        assert all(float(np.float32(x)) == x for x in got.values())
    assert bja["bidirectional"] == {"constant": [S + 1, S + 2], "variable": [S + 2, S + 10]}
    assert bja["phase_counts"]["4"] == {"survivor_rows": 3, "distinct_survivors": 3}
    assert bja["credit"] == {"granted": True, "certified_interval": [S, S + N],
                             "linked_interval": [S, S + N],
                             "swept_intervals": {p: [S, S + N] for p in ("1", "2", "3", "4")}}
    assert bja["inputs"]["run.dataset_sha256"] == DATASET_SHA
    assert bja["inputs"]["param.warm_start_skip_max"] == 16
    assert "use_persistent_workers" in bja["geometry"]
    assert X._sha256(os.path.join(os.path.dirname(ba), "survivors.npz")) == art.artifact_sha256
    assert bja["provenance"]["pwc_completion_rebuilt"]["trial_complete"] is True
    return f"bundles SAME (exit 0); {len(rep['outputs'])} output rows; NPZ {art.final_row_count} rows"


def g_readonly():
    r, p, _ = make_tree("ro")
    before = tree_digest(r)
    extract("ro", r, p)
    after = tree_digest(r)
    assert before == after, sorted(set(before.items()) ^ set(after.items()))[:4]
    side = [f for f in os.listdir(r) if f.startswith("prng_analysis.db-")]
    assert not side, side
    return f"{len(before)} run-tree files unchanged; no ledger sidecar"


def g_summary():
    ra, pa, _ = make_tree("sum_a", summary=("4",))
    rb, pb, _ = make_tree("sum_b", summary=("4",))
    rf, pf, _ = make_tree("sum_full")
    ba, bja = extract("sum_a", ra, pa)
    bb, _ = extract("sum_b", rb, pb)
    bf, _ = extract("sum_full", rf, pf)
    assert bja["phases"]["4"] is None and bja["bidirectional"]["variable"] is None
    assert bja["phase_counts"]["4"] == {"survivor_rows": 3, "distinct_survivors": None}
    assert any("NOT RUN" in c for c in bja["provenance"]["self_checks"])
    rc, rep, _ = cmp(ba, bb)
    assert rc == 3 and "phases.4" in rep["not_compared"] and "bidirectional.variable" in rep["not_compared"], (rc, rep["not_compared"])
    rc2, rep2, _ = cmp(bf, ba)
    assert rc2 == 3 and "phases.4" in rep2["not_compared"], (rc2, rep2["not_compared"])
    assert not any(r["status"] == CEO.DIFFERENT for r in rep2["outputs"])
    return "summary-only phase: null; vs summary or full bundle exit 3, phases.4 NOT COMPARED"


def refusal(tag, must, build=None, **ekw):
    r, p, _ = (build or (lambda: make_tree(tag)))()
    try:
        extract(tag, r, p, **ekw)
    except X.ExtractRefused as exc:
        assert must in str(exc), f"{tag}: refused, but for {exc}"
        assert not os.path.exists(os.path.join(OUT, tag, "bundle", "bundle.json"))
        return f"refused: {str(exc)[:90]}"
    raise AssertionError(f"{tag}: not refused")


def r2_hash():
    r, p, art = make_tree("r2")
    g = os.path.join(OUT, "r2", "gen_copy")
    shutil.copytree(art.generation_dir, g)
    with open(os.path.join(g, X.BINARY_NPZ), "ab") as f:
        f.write(b"\0")
    try:
        extract("r2", r, p, generation_dir=g)
    except X.ExtractRefused as exc:
        assert "sha256" in str(exc), exc
        return f"refused: {str(exc)[:90]}"
    raise AssertionError("tampered NPZ accepted")


def r7_float64():
    def build():
        r, p, a = make_tree("r7")
        f = X.pass_file(os.path.join(r, "results"), "1", WS, OFF, ORD)
        d = json.load(open(f))
        d["match_rates"][0] = 0.71                         # a float64, not a float32
        st = os.stat(f)
        json.dump(d, open(f, "w"))
        os.utime(f, (st.st_atime, st.st_mtime))
        return r, p, a
    return refusal("r7", "exact float32", build)


def r11_stale():
    def build():
        r, p, a = make_tree("r11")
        f = X.pass_file(os.path.join(r, "results"), "2", WS, OFF, ORD)
        t = time.time() + 60
        os.utime(f, (t, t))
        return r, p, a
    return refusal("r11", "newer", build)


def r12_out_nonempty():
    r, p, _ = make_tree("r12")
    os.makedirs(os.path.join(OUT, "r12", "bundle"))
    open(os.path.join(OUT, "r12", "bundle", "x"), "w").close()
    try:
        extract("r12", r, p)
    except X.ExtractRefused as exc:
        assert "never overwritten" in str(exc)
        return "refused: out dir not empty"
    raise AssertionError("overwrote a non-empty out dir")


def r8_union():
    bad = copy.deepcopy(PHASE_DATA)
    bad["2"] = [(2, 0.8333333), (7, 0.75)]           # seed 1 leaves the constant set
    return refusal("r8", "constant bidirectional set",
                   lambda: make_tree("r8", phase_data=bad, cand_data=PHASE_DATA))


# ---------------------------------------------------------------- Beta X-PWC-STAGE4 R1
def _ledger_rows(r):
    con = sqlite3.connect(f"file:{os.path.join(r, 'prng_analysis.db')}?mode=ro&immutable=1", uri=True)
    try:
        cov = con.execute("SELECT * FROM certified_coverage").fetchall()
        link = con.execute(f"SELECT * FROM {T_LINK}").fetchall()
    finally:
        con.close()
    return cov, link


def _rebuild_ledger(r, cov_rows, link_rows, wal_only=False):
    """Replace the run ledger. With wal_only, the rows live ONLY in a non-empty -wal next
    to a main file that lacks them (a ledger left with an uncheckpointed WAL)."""
    db = os.path.join(r, "prng_analysis.db")
    for f in (db, db + "-wal", db + "-shm"):
        if os.path.exists(f):
            os.remove(f)
    if not wal_only:
        con = sqlite3.connect(db)
        con.executescript(COV_DDL); con.execute(S1_DDL[T_LINK])
        con.executemany("INSERT INTO certified_coverage VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", cov_rows)
        con.executemany(f"INSERT INTO {T_LINK} VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", link_rows)
        con.commit(); con.close()
        return
    stage = os.path.join(os.path.dirname(r), "wal_stage")
    os.makedirs(stage, exist_ok=True)
    sdb = os.path.join(stage, "prng_analysis.db")
    con = sqlite3.connect(sdb)
    con.execute("PRAGMA journal_mode=WAL"); con.execute("PRAGMA wal_autocheckpoint=0")
    con.executescript(COV_DDL); con.execute(S1_DDL[T_LINK]); con.commit()
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")          # schema in the main file
    con.executemany("INSERT INTO certified_coverage VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", cov_rows)
    con.executemany(f"INSERT INTO {T_LINK} VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", link_rows)
    con.commit()                                           # rows only in the -wal
    shutil.copyfile(sdb, db); shutil.copyfile(sdb + "-wal", db + "-wal")   # copied while open
    con.close()


def g_wal_only_credit():
    r, p, art = make_tree("wal")
    cov, link = _ledger_rows(r)
    _rebuild_ledger(r, cov, link, wal_only=True)
    db = os.path.join(r, "prng_analysis.db")
    assert os.path.getsize(db + "-wal") > 0
    c = sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)
    n_main = c.execute("SELECT count(*) FROM certified_coverage").fetchone()[0]; c.close()
    assert n_main == 0, "fixture: the main file must lack the rows (else not WAL-only)"
    before = tree_digest(r)
    _, b = extract("wal", r, p)
    assert b["credit"]["granted"] is True and b["credit"]["linked_interval"] == [S, S + N], b["credit"]
    assert b["inputs"]["run.dataset_sha256"] == DATASET_SHA
    assert b["provenance"]["ledger_read"] == "wal_copy"
    assert tree_digest(r) == before and not os.path.exists(db + "-shm"), "run tree changed"
    assert not [f for f in os.listdir(os.path.join(OUT, "wal", "bundle")) if f.startswith(".extract_sqlite_")]
    return (f"rows only in a {os.path.getsize(db + '-wal')}-byte WAL: credit read from a stable "
            f"copy (an immutable read sees {n_main} rows); run tree unchanged, no -shm")


def g_journal_unavailable():
    r, p, _ = make_tree("jrnl")
    with open(os.path.join(r, "prng_analysis.db-journal"), "wb") as f:
        f.write(b"\0" * 512)
    try:
        extract("jrnl", r, p)
    except X.ExtractRefused as exc:
        assert "UNAVAILABLE" in str(exc), exc
        assert not os.path.exists(os.path.join(OUT, "jrnl", "bundle", "bundle.json"))
        return f"refused: {str(exc)[:80]}"
    raise AssertionError("non-empty -journal accepted")


def _cross(tag, mutate, must):
    def build():
        r, p, a = make_tree(tag)
        cov, link = _ledger_rows(r)
        cov, link = mutate([list(x) for x in cov], [list(x) for x in link], a)
        _rebuild_ledger(r, [tuple(x) for x in cov], [tuple(x) for x in link])
        return r, p, a
    return refusal(tag, must, build)


def x1(cov, link, a):          # coverage: same artifact, another generation
    cov[0][11] = "gen-other"; return cov, link


def x2(cov, link, a):          # coverage: this generation, another artifact
    cov[0][10] = "f" * 64; return cov, link


def x3(cov, link, a):          # a SECOND generation's row sharing this artifact hash
    extra = list(cov[0]); extra[0] = "cov-other"; extra[11] = "gen-other"
    return cov + [extra], link


def x4(cov, link, a):          # link names another coverage id
    link[0][1] = "cov-other"; return cov, link


def x5(cov, link, a):          # link interval differs
    link[0][7] = S + N - 1; return cov, link


def x6(cov, link, a):          # no link row
    return cov, []


def g_no_link_table():
    def build():
        r, p, a = make_tree("nolinktab")
        cov, _ = _ledger_rows(r)
        db = os.path.join(r, "prng_analysis.db"); os.remove(db)
        con = sqlite3.connect(db); con.executescript(COV_DDL)
        con.executemany("INSERT INTO certified_coverage VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", cov)
        con.commit(); con.close()
        return r, p, a
    return refusal("nolinktab", "no step1_chunk_link table", build)


def g_out_inside_run():
    r, p, _ = make_tree("outin")
    inside = os.path.join(r, "bundle_here")
    try:
        X.extract(r, p, inside, live_paths=())
    except X.ExtractRefused as exc:
        assert "inside the run tree" in str(exc), exc
        assert not os.path.exists(inside)
        return "refused: --out-dir inside the run tree; nothing created"
    raise AssertionError("wrote a bundle inside the run tree")


def g_membership_one_mode():
    """Phase 4 summary-only (variable set unavailable), and an NPZ constant row outside the
    constant set: still refused (before, a summary in one mode disabled all checks)."""
    bad = copy.deepcopy(PHASE_DATA)
    bad["2"] = [(2, 0.8333333), (7, 0.75)]
    return refusal("m1mode", "constant bidirectional set", lambda: make_tree(
        "m1mode", phase_data=bad, cand_data=PHASE_DATA, summary=("4",)))


def g_ordinal_is_t1():
    """The real Stage 4 case: files at _t1, credit says trial 0. Extracted, ordinal recorded."""
    r, p, _ = make_tree("ord1")
    assert os.path.exists(X.pass_file(os.path.join(r, "results"), "1", WS, OFF, 1))
    assert not os.path.exists(X.pass_file(os.path.join(r, "results"), "1", WS, OFF, 0))
    _, b = extract("ord1", r, p)
    assert b["provenance"]["record_ordinal"] == 1, b["provenance"]["record_ordinal"]
    assert any("exactly one complete trial" in x for x in b["provenance"]["record_ordinal_binding"])
    return "files at _t1 (no _t0): extracted, record ordinal 1 bound by link and NPZ"


def g_ordinal_not_guessed():
    """Any single shared ordinal is discovered, e.g. a resumed run's _t5."""
    r, p, _ = make_tree("ord5", ordinals={ph: [5] for ph in ("1", "2", "3", "4")}, cand_ord=5)
    _, b = extract("ord5", r, p)
    assert b["provenance"]["record_ordinal"] == 5
    return "files at _t5, NPZ rows trial_number 5: discovered 5, nothing assumed"


def g_run_log():
    r, p, _ = make_tree("rlog")
    log = os.path.join(OUT, "rlog", "pwc_run.log")
    with open(log, "w") as f:
        for stem in ("forward", "reverse", "forward_hybrid", "reverse_hybrid"):
            f.write(f"2026-11-21 13:47:19,855 - PersistentWorkerCoordinator - INFO - Results saved: "
                    f"results/window_opt_{stem}_{WS}_{OFF}_t1.json (0 survivors)\n")
    _, b = X.extract(r, p, os.path.join(OUT, "rlog", "bundle"), live_paths=(), run_log=log)
    assert any("run log records" in x for x in b["provenance"]["record_ordinal_binding"])
    log2 = os.path.join(OUT, "rlog", "pwc_run_short.log")
    open(log2, "w").write(open(log).read().replace("forward_hybrid", "XX"))
    try:
        X.extract(r, p, os.path.join(OUT, "rlog", "bundle2"), live_paths=(), run_log=log2)
    except X.ExtractRefused as exc:
        assert "does not record saving" in str(exc), exc
        return "run log naming all four files: accepted; one missing: refused"
    raise AssertionError("a run log missing a pass file was accepted")


def g_symlink_live_sidecar():
    """[Beta X-PWC-STAGE4 R2 Q5] provenance.json is a symlink to a file under a (stand-in)
    live root: refused BEFORE the file is opened. The target is deliberately not JSON, so a
    read before the guard would fail with 'cannot read provenance' instead."""
    r, p, art = make_tree("symlive")
    live = os.path.join(OUT, "symlive", "live_root")
    os.makedirs(live)
    target = os.path.join(live, "provenance.json")
    open(target, "w").write("{not json - must never be read")
    g = os.path.join(OUT, "symlive", "gen_copy")
    shutil.copytree(art.generation_dir, g)
    os.remove(os.path.join(g, "provenance.json"))
    os.symlink(target, os.path.join(g, "provenance.json"))
    try:
        X.extract(r, p, os.path.join(OUT, "symlive", "bundle"), generation_dir=g,
                  live_paths=(live,))
    except X.ExtractRefused as exc:
        assert "live path" in str(exc) and "provenance" in str(exc), exc
        return "symlinked sidecar into a live root: refused before any read"
    raise AssertionError("followed a symlink into a live root")


ZERO = {"1": [(1, 0.75)], "2": [(2, 0.75)], "3": [(3, 0.75)], "4": [(4, 0.75)]}   # no intersection


def write_log(path, names, prefix="results/"):
    with open(path, "w") as f:
        for n in names:
            f.write(f"2026-11-21 13:47:19,855 - PersistentWorkerCoordinator - INFO - Results saved: "
                    f"{prefix}{n} (1 survivors)\n")
    return path


def names(o=1):
    return [f"window_opt_{st}_{WS}_{OFF}_t{o}.json"
            for st in ("forward", "reverse", "forward_hybrid", "reverse_hybrid")]


def g_zero_rows_needs_log():
    r, p, _ = make_tree("zero1", phase_data=ZERO)
    try:
        X.extract(r, p, os.path.join(OUT, "zero1", "bundle"), live_paths=())
    except X.ExtractRefused as exc:
        assert "--run-log" in str(exc) and "zero-row" in str(exc), exc
    else:
        raise AssertionError("a zero-row bundle was extracted without the run log")
    log = write_log(os.path.join(OUT, "zero1", "run.log"), names())
    _, b = X.extract(r, p, os.path.join(OUT, "zero1", "bundle2"), live_paths=(), run_log=log)
    bind = b["provenance"]["record_ordinal_binding"]
    assert any("UNAVAILABLE (no rows)" in x for x in bind) and b["provenance"]["run_log"]["sha256"]
    return "zero rows: refused without --run-log; with it, bound by the log (NPZ check reported UNAVAILABLE)"


def g_beta_t5_case():
    """Beta's reproduction: empty NPZ, all four files renamed _t1 -> _t5 and a phase-3
    survivor added; the stopped run's ORIGINAL log (naming _t1) must reject it."""
    r, p, _ = make_tree("t5", phase_data=ZERO)
    log = write_log(os.path.join(OUT, "t5", "run.log"), names(1))
    res = os.path.join(r, "results")
    for ph in ("1", "2", "3", "4"):
        a, b_ = X.pass_file(res, ph, WS, OFF, 1), X.pass_file(res, ph, WS, OFF, 5)
        st = os.stat(a)
        d = json.load(open(a))
        if ph == "3":
            d["survivors"].append(S + 9); d["match_rates"].append(F32(0.75)); d["survivor_count"] += 1
        json.dump(d, open(b_, "w")); os.remove(a); os.utime(b_, (st.st_atime, st.st_mtime))
    try:
        X.extract(r, p, os.path.join(OUT, "t5", "bundle"), live_paths=(), run_log=log)
    except X.ExtractRefused as exc:
        assert "does not record saving" in str(exc), exc
        return "renamed _t5 set + added survivor, empty NPZ: rejected by the run's own log"
    raise AssertionError("a foreign pass set was accepted")


def g_log_substring():
    r, p, _ = make_tree("subs")
    log = write_log(os.path.join(OUT, "subs", "run.log"),
                    [f"other-{n}.bak" for n in names()])
    try:
        X.extract(r, p, os.path.join(OUT, "subs", "bundle"), live_paths=(), run_log=log)
    except X.ExtractRefused as exc:
        assert "exact path match" in str(exc), exc
        return "log naming results/other-<name>.bak: refused (exact path, not substring)"
    raise AssertionError("a substring match was accepted")


def g_run_log_symlink_live():
    r, p, _ = make_tree("logsym")
    live = os.path.join(OUT, "logsym", "live_root"); os.makedirs(live)
    target = write_log(os.path.join(live, "pwc_run.log"), names())
    link = os.path.join(OUT, "logsym", "run.log")
    os.symlink(target, link)
    try:
        X.extract(r, p, os.path.join(OUT, "logsym", "bundle"), live_paths=(live,), run_log=link)
    except X.ExtractRefused as exc:
        assert "--run-log" in str(exc) and "live path" in str(exc), exc
        return "--run-log symlinked into a live root: refused before it is opened"
    raise AssertionError("followed a --run-log symlink into a live root")


GATES_ORDINAL = [
    ("Q5-ZERO-ROWS-REQUIRE-RUN-LOG", g_zero_rows_needs_log),
    ("Q5-BETA-T5-CASE", g_beta_t5_case),
    ("Q5-LOG-EXACT-PATH (no substring)", g_log_substring),
    ("F2-RUN-LOG-SYMLINK-LIVE", g_run_log_symlink_live),
    ("B2-SYMLINK-LIVE-SIDECAR (Beta R2-Q5)", g_symlink_live_sidecar),
    ("O-ORDINAL-T1 (the Stage 4 run's case)", g_ordinal_is_t1),
    ("O-ORDINAL-DISCOVERED-NOT-GUESSED (_t5)", g_ordinal_not_guessed),
    ("O-ORDINAL-AMBIGUOUS (_t1 and _t2 sets)", lambda: refusal(
        "oamb", "not one shared value", lambda: make_tree(
            "oamb", ordinals={ph: [1, 2] for ph in ("1", "2", "3", "4")}))),
    ("O-ORDINAL-SPLIT (phase 3 at _t2)", lambda: refusal(
        "osplit", "not one shared value", lambda: make_tree("osplit", ordinals={"3": [2]}))),
    ("O-ORDINAL-ASSERTED-MISMATCH", lambda: refusal("oassert", "asserted", None, record_ordinal=0)),
    ("O-NPZ-ORDINAL-MISMATCH (rows trial 0, files _t1)", lambda: refusal(
        "onpz", "not the same trial", lambda: make_tree("onpz", cand_ord=0))),
    ("O-LINK-TWO-TRIALS", lambda: refusal(
        "olink", "not exactly one complete trial", lambda: make_tree("olink", link_trials=(2, 0, 0, 0)))),
    ("O-RUN-LOG", g_run_log),
]


GATES_BETA = [
    ("B-WAL-ONLY-CREDIT", g_wal_only_credit),
    ("B-JOURNAL-UNAVAILABLE", g_journal_unavailable),
    ("B-X1 coverage of another generation, same artifact", lambda: _cross("x1", x1, "certified_coverage holds")),
    ("B-X2 coverage of this generation, another artifact", lambda: _cross("x2", x2, "certified_coverage holds")),
    ("B-X3 a second generation's row shares the artifact", lambda: _cross("x3", x3, "certified_coverage holds")),
    ("B-X4 link names another coverage id", lambda: _cross("x4", x4, "the link names coverage")),
    ("B-X5 link interval differs", lambda: _cross("x5", x5, "linked interval")),
    ("B-X6 no link row", lambda: _cross("x6", x6, "step1_chunk_link rows")),
    ("B-NO-LINK-TABLE", g_no_link_table),
    ("B-THRESHOLD-UNVERIFIED (phase 2)", lambda: refusal("thr1", "threshold record not verified", lambda: make_tree(
        "thr1", pass_over={"2": {"threshold": threshold_rec("2", verified=False)}}))),
    ("B-THRESHOLD-WRONG-REQUEST (phase 3)", lambda: refusal("thr2", "requested", lambda: make_tree(
        "thr2", pass_over={"3": {"threshold": threshold_rec("3", requested=0.5)}}))),
    ("B-ACCOUNTING completed < planned, complete still true", lambda: refusal(
        "acc1", "completion conditions", lambda: make_tree(
            "acc1", pass_over={"1": {"completed": 3, "failed": 1}}))),
    ("B-ACCOUNTING completed_intervals leave a gap", lambda: refusal(
        "acc2", "completed_intervals", lambda: make_tree(
            "acc2", pass_over={"2": {"completed_intervals": [[S, S + 500], [S + 600, S + N]]}}))),
    ("B-OUT-DIR-INSIDE-RUN", g_out_inside_run),
    ("B-MEMBERSHIP-ONE-MODE-AVAILABLE", g_membership_one_mode),
]


def g_subprocess():
    r, p, _ = make_tree("sub")
    out = os.path.join(OUT, "sub", "bundle")
    res = subprocess.run([sys.executable, TOOL, "--run-dir", r, "--params-json", p,
                          "--out-dir", out], capture_output=True, text=True, timeout=120)
    assert res.returncode == 0 and "BUNDLE_WRITTEN" in res.stdout, (res.returncode, res.stderr[-400:])
    res = subprocess.run([sys.executable, TOOL, "--run-dir", r, "--params-json", p,
                          "--out-dir", out], capture_output=True, text=True, timeout=120)
    assert res.returncode == 2 and "REFUSED" in res.stderr, (res.returncode, res.stderr[-300:])
    return "as a program: exit 0 writes, exit 2 refuses (second run, out dir not empty)"


GATES = [
    ("G-CLEAN", g_clean),
    ("G-READONLY", g_readonly),
    ("G-SUMMARY", g_summary),
    ("R1 generation with a parent", lambda: refusal(
        "r1", "has a parent", lambda: make_tree("r1", twice=True))),
    ("R2 NPZ hash mismatch", r2_hash),
    ("R3 pass file missing", lambda: refusal("r3", "no pass file", lambda: (
        lambda t: (os.remove(X.pass_file(os.path.join(t[0], "results"), "3", WS, OFF, ORD)), t)[1])(
        make_tree("r3")))),
    ("R4 pass not on TCP", lambda: refusal(
        "r4", "not tcp", lambda: make_tree("r4", pass_over={"2": {"transport": "ssh"}}))),
    ("R5 interval mismatch", lambda: refusal(
        "r5", "requested_interval", lambda: make_tree(
            "r5", pass_over={"1": {"requested_interval": [S, S + N - 1]}}))),
    ("R6 incomplete pass", lambda: refusal(
        "r6", "completion conditions", lambda: make_tree("r6", pass_over={"4": {"complete": False}}))),
    ("R7 float64 match rate", r7_float64),
    ("R8 NPZ seeds != union of sets", r8_union),
    ("R9 not a PWC run", lambda: refusal("r9", "not a PWC run", lambda: make_tree(
        "r9", params=dict(PARAMS, use_range_miner=True, use_persistent_workers=False)))),
    ("R10 live path", lambda: refusal("r10", "live path", None,
                                      live_paths=(os.path.join(OUT, "r10"),))),
    ("R11 stale pass file", r11_stale),
    ("R12 non-empty out dir", r12_out_nonempty),
    ("R13 no ledger", lambda: refusal("r13", "no run ledger", lambda: make_tree("r13", ledger=False))),
    *GATES_BETA,
    *GATES_ORDINAL,
    ("G-SUBPROCESS", g_subprocess),
]


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    print("VERIFICATION test_extract_pwc_bundle START", flush=True)
    passed = 0
    for name, fn in GATES:
        try:
            print(f"[PASS] {name}: {fn()}", flush=True)
            passed += 1
        except Exception as exc:                                    # noqa: BLE001
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}", flush=True)
    print(f"\n{passed}/{len(GATES)} PASS")
    print("RESULT: PASS" if passed == len(GATES) else "RESULT: FAIL")
    print("VERIFICATION test_extract_pwc_bundle COMPLETE", flush=True)
    return 0 if passed == len(GATES) else 1


if __name__ == "__main__":
    sys.exit(main())
