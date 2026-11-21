#!/usr/bin/env python3
"""test_extract_miner_bundle.py: scripts/extract_miner_bundle.py on synthetic finished
RANGE-MINER runs. CPU only: no GPU, no rig, no live ledger. Files go under
<root>/logs/test_extract_miner_bundle/ (ignored).

Run from the repository root:   python3 -u tests/test_extract_miner_bundle.py

It needs the Stage 4 comparator (`scripts/compare_engine_outputs.py` with `build_inputs`,
the null rule and exit 3); G-COMPARATOR checks that first.

A synthetic run is built with the REAL writers: the ledger schema (`MinerLedger`), the
finalizer (`finalize_run` over records from `build_mode_records`), the certification door
(`CoverageLedger.record_publication`) and the search link DDL (`step1_search_ledger`), so
the extractor reads the production layout, not a copy of it.

Beta X-PWC-STAGE4 R1 regressions: G-MIXED-SOURCE (Q6), G-COVERAGE-CROSS-ID (Q3),
G-NON-DONE, G-PHASE-SET, G-TRIAL-ORDINAL (Q9), and one gate per binding check B1-B8.
Mutants: each is one change to the extractor's source; each must make a named gate fail.
"""
import contextlib
import gc
import hashlib
import io
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import types
import zipfile
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
OUT = os.path.join(ROOT, "logs", "test_extract_miner_bundle")
TOOL = os.path.join(ROOT, "scripts", "extract_miner_bundle.py")
CMP = os.path.join(ROOT, "scripts", "compare_engine_outputs.py")

import importlib.util                                                    # noqa: E402
from miner.range_miner_coordinator import MinerLedger                    # noqa: E402
from utils.canonical_records import build_mode_records                   # noqa: E402
from utils.run_finalizer import finalize_run                             # noqa: E402
from utils.seed_coverage_ledger import CoverageLedger                    # noqa: E402
import utils.step1_search_ledger as SSL                                  # noqa: E402

_spec = importlib.util.spec_from_file_location("compare_engine_outputs", CMP)
CEO = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CEO)

TOOL_SRC = Path(TOOL).read_text(encoding="utf-8")


def load_tool(src=TOOL_SRC):
    """The extractor as a module. A mutant keeps __file__ = the real path, so its
    imports and code root are the real tree's."""
    mod = types.ModuleType("extract_miner_bundle")
    mod.__file__ = TOOL
    exec(compile(src, TOOL, "exec"), mod.__dict__)
    return mod


EXT = None
RUN_ID = "cfg_t0_deadbeef"
TRIAL = 0
S0, N = 1000, 400                       # interval [1000, 1400), 2 stripes x 2 shards per phase
MAPS = {1: {1001: 0.5, 1002: 0.625, 1100: 0.75, 1300: 0.5},       # forward constant
        2: {1002: 0.5, 1100: 0.875, 1399: 0.5},                   # reverse constant
        3: {1100: 0.625, 1250: 0.5},                              # forward variable
        4: {1100: 0.75, 1250: 0.5, 1251: 0.5}}                    # reverse variable
CTX = {"window_size": 8, "window_anchor": 0, "skip_min": 6, "skip_max": 99,
       "sessions": ["evening"], "trial_number": TRIAL, "prng_base": "java_lcg"}
DSHA = "ab" * 32
FT, RT = 0.71, 0.47
PINS = {"warm_start_window": 8, "warm_start_offset": 0, "warm_start_skip_min": 6,
        "warm_start_session_idx": 1,
        "warm_start_skip_max": 99, "warm_start_fwd_thresh": FT, "warm_start_rev_thresh": RT}
NO_LIVE = ()                            # live-path refusal is tested by G-LIVE-PATH
TRIAL_SECONDS = 60.0                    # synthetic trial: finished 1 s before publication


def params_for(staging, **over):
    p = {"use_range_miner": True, "use_persistent_workers": False, "window_trials": 1,
         "prng_type": "java_lcg", "test_both_modes": True, "seed_start": S0, "max_seeds": N,
         "worker_pool_size": 8, "staging_dir": staging, **PINS}
    p.update(over)
    return {k: v for k, v in p.items() if v is not DROP}


DROP = object()


# --------------------------------------------------------------------------- builders
def _checkpoint(db):
    gc.collect()
    con = sqlite3.connect(db)
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    con.close()


def build_ledger(staging, run_id=RUN_ID, trial=TRIAL, state="committed", phases=(1, 2, 3, 4),
                 extra_trials=(), mutate=None, times=None):
    os.makedirs(staging, exist_ok=True)
    db = os.path.join(staging, "miner_ledger.db")
    MinerLedger(db)
    now = time.time()
    created, finalized = times or (now - 1 - TRIAL_SECONDS, now - 1)
    con = sqlite3.connect(db)
    for rid, tn, st in ((run_id, trial, state),) + tuple(extra_trials):
        con.execute("INSERT INTO trials (run_id, trial_number, state, commit_delivery_status, "
                    "created_at, finalized_at) VALUES (?,?,?,?,?,?)",
                    (rid, tn, st, "done" if st == "committed" else "none", created, finalized))
        con.execute("INSERT INTO trial_context VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (rid, tn, CTX["window_size"], CTX["window_anchor"], 0,
                     json.dumps(CTX["sessions"]), CTX["skip_min"], CTX["skip_max"], "java_lcg",
                     FT, RT, DSHA, "cd" * 32, created))
    for ph in phases:
        total = len(MAPS[ph])
        for k in range(2):
            sid = f"p{ph}_s{k}"
            st_start, st_n = S0 + k * 200, 200
            surv = total if k == 0 else 0
            con.execute("INSERT INTO stripes (run_id, stripe_id, seed_start, seed_count, state, "
                        "current_attempt, phase, family_name, survivors_total, created_at) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (run_id, sid, st_start, st_n, "done", 0, ph, "java_lcg", surv, created))
            for sub in range(2):
                con.execute("INSERT INTO shards (run_id, stripe_id, attempt, sub_index, worker_id, "
                            "seed_start, seed_count, survivor_count, staging_status, created_at, "
                            "local_cleanup_status) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                            (run_id, sid, 0, sub, f"rig{sub}:gpu{k}", st_start + sub * 100, 100,
                             surv if sub == 0 else 0, "verified", created, "deleted"))
    if mutate:
        mutate(con)
    con.commit()
    con.close()
    _checkpoint(db)
    return db


def build_generation(root, prior=False, certify=True, link=True, maps=None, ctx=None,
                     seed_count=N):
    maps, ctx = maps or MAPS, ctx or CTX
    os.makedirs(root, exist_ok=True)
    cands = []
    for mode, (f, r), pt in (("constant", (1, 2), "java_lcg"), ("variable", (3, 4), "java_lcg_hybrid")):
        _, recs = build_mode_records(maps[f], maps[r], ctx, mode, pt)
        cands += recs
    kw = dict(output_root=Path(root), run_id="step1_run", prng_base="java_lcg",
              skip_modes_executed=("constant", "variable"), seed_start=S0, seed_count=seed_count,
              repository_commit="0" * 40, repository_tree_clean=True)
    res = finalize_run(cands, **kw)
    if prior:
        res = finalize_run(cands, **kw)          # a second generation, parent = the first
    db = os.path.join(root, "prng_analysis.db")
    cl = CoverageLedger(db)
    cl.init_schema()
    if certify:
        cl.record_publication(res, dataset_sha256=DSHA)
    if link:
        con = sqlite3.connect(db)
        con.execute(SSL._DDL[SSL.T_LINK])
        if certify:
            cov_id = con.execute("SELECT coverage_id FROM certified_coverage").fetchone()[0]
            con.execute(f"INSERT INTO {SSL.T_LINK} VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        ("l1", cov_id, "srch1", "chunk1", "java_lcg", '["constant","variable"]',
                         S0, S0 + seed_count, res.generation_id, 1, 0, 0, 0,
                         "2026-11-21T00:00:00Z"))
        con.commit()
        con.close()
    _checkpoint(db)
    return res


def build_run(tag, **kw):
    base = os.path.join(OUT, tag)
    if os.path.isdir(base):            # a gate run again (mutants) starts from nothing
        shutil.rmtree(base)
    staging, outroot = os.path.join(base, "staging"), os.path.join(base, "outroot")
    gen_kw = {k: kw.pop(k) for k in ("prior", "certify", "link", "maps", "ctx", "seed_count")
              if k in kw}
    pover = kw.pop("params", {})
    build_ledger(staging, **kw)
    res = build_generation(outroot, **gen_kw)
    pj = os.path.join(base, "params.json")
    Path(pj).write_text(json.dumps(params_for(staging, **pover)))
    return base, staging, outroot, pj, res


def snapshot(*dirs):
    snap = {}
    for d in dirs:
        for dp, dns, fs in os.walk(d):
            for fn in fs + dns:
                p = os.path.join(dp, fn)
                if os.path.islink(p):
                    snap[p] = ("link", os.readlink(p))
                elif os.path.isfile(p):
                    st = os.stat(p)
                    snap[p] = (st.st_size, st.st_mtime_ns,
                               hashlib.sha256(Path(p).read_bytes()).hexdigest())
    return snap


def receipt(run, name="receipt", **kw):
    base, staging, outroot, pj, _ = run
    kw.setdefault("live_paths", NO_LIVE)
    path = os.path.join(base, f"{name}.json")
    return EXT.write_receipt(staging, outroot, pj, path, **kw)[0]


def bundle(run, rcpt, name="bundle", **kw):
    base, staging, outroot, pj, _ = run
    kw.setdefault("live_paths", NO_LIVE)
    return EXT.build_bundle(staging, outroot, pj, os.path.join(base, name), rcpt, **kw)


def extract(run, name="bundle", **kw):
    """Both steps, as the Stage 4 session runs them."""
    rk = {k: kw[k] for k in ("run_id", "ledger_db", "live_paths", "code_root") if k in kw}
    return bundle(run, receipt(run, name=f"{name}_receipt", **rk), name=name, **kw)


def refused(fn, needle):
    try:
        fn()
    except EXT.ExtractRefused as exc:
        assert needle in str(exc), f"refused, but for {exc!s:.300}"
        return str(exc)
    raise AssertionError(f"not refused (wanted {needle!r})")


def compare(pa, pb, require_credit=True):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = CEO.main([pa, pb] + ([] if require_credit else ["--no-require-credit"]))
    rep = CEO.compare(CEO.load_bundle(pa), CEO.load_bundle(pb), require_credit=require_credit)
    return rc, rep


def status(rep):
    return {r["field"]: r["status"] for r in rep["inputs"] + rep["outputs"]}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def set_trial_times(staging, created, finalized):
    db = os.path.join(staging, "miner_ledger.db")
    con = sqlite3.connect(db)
    con.execute("UPDATE trials SET created_at=?, finalized_at=?", (created, finalized))
    con.commit()
    con.close()
    _checkpoint(db)


# --------------------------------------------------------------------------- gates
def g_comparator():
    for name in ("build_inputs", "exact_float32", "exit_code", "NOT_COMPARED"):
        assert hasattr(CEO, name), f"the comparator at {CMP} has no {name}: not the Stage 4 one"
    return f"comparator has build_inputs, exact_float32, exit_code, NOT COMPARED (sha256 {sha(CMP)[:16]})"


def g_clean():
    run = build_run("clean")
    base, staging, outroot, pj, res = run
    before = snapshot(staging, outroot)
    out, b = extract(run)
    assert snapshot(staging, outroot) == before, "the extractor changed the run's directories"
    CEO.load_bundle(out)
    npz = os.path.join(os.path.dirname(out), "survivors.npz")
    assert Path(npz).read_bytes() == Path(res.binary_npz_path).read_bytes(), "NPZ not byte-identical"
    assert b["phases"] == {p: None for p in "1234"}, b["phases"]
    assert b["bidirectional"] == {"constant": None, "variable": None}, b["bidirectional"]
    assert b["phase_counts"] == {str(p): {"survivor_rows": len(MAPS[p]), "distinct_survivors": None}
                                 for p in MAPS}, b["phase_counts"]
    assert b["credit"] == {"granted": True, "certified_interval": [S0, S0 + N],
                           "linked_interval": [S0, S0 + N],
                           "swept_intervals": {str(p): [S0, S0 + N] for p in MAPS}}, b["credit"]
    prov = json.loads(Path(res.sidecar_path).read_text())
    want_inputs, want_geo = CEO.build_inputs(json.loads(Path(pj).read_text()), prov, DSHA)
    assert b["inputs"] == want_inputs, (b["inputs"], want_inputs)
    assert all(b["geometry"][k] == v for k, v in want_geo.items())
    assert b["search_id"] == "srch1"
    na = {e["piece"] for e in b["provenance"]["not_available"]}
    assert na == ({f"phases.{p}" for p in "1234"} | {"bidirectional.constant", "bidirectional.variable"}
                  | {f"phase_counts.{p}.distinct_survivors" for p in "1234"}), sorted(na)
    j = b["provenance"]["job_identity"]
    assert (j["seed_range"], j["window_size"], j["window_anchor"], j["sessions"], j["dataset_sha256"],
            j["generator_phase"], j["forward_threshold"], j["reverse_threshold"],
            j["constant_skip_range_requested"], j["constant_skip_range_effective"], j["phases"],
            j["prng_family"]) == ([S0, S0 + N], 8, 0, ["evening"], DSHA, 0, FT, RT, [6, 99], [0, 16],
                                  [1, 2, 3, 4], "java_lcg"), j
    assert b["provenance"]["receipt_sha256"]
    return ("receipt + bundle; NPZ byte-identical; inputs == build_inputs(); 10 nulls; counts, "
            "credit, F2 job identity exact; run dirs unchanged")


def g_null_never_equal():
    run = build_run("never_equal")
    pa, _ = extract(run, name="A")
    pb, _ = extract(run, name="B")
    rc, rep = compare(pa, pb)
    st = status(rep)
    nulls = ({f"phases.{p}" for p in "1234"} | {"bidirectional.constant", "bidirectional.variable"}
             | {f"phase_counts.{p}.distinct_survivors" for p in "1234"})
    for f in nulls:
        assert st.get(f) == CEO.NOT_COMPARED, (f, st.get(f))
    assert rc == 3 and rep["verdict"] == "SAME WHERE COMPARED", (rc, rep["verdict"])
    others = {f: s for f, s in st.items() if f not in nulls}
    assert all(s == CEO.EQUAL for s in others.values()), {f: s for f, s in others.items() if s != CEO.EQUAL}
    assert sum(f.startswith("npz.") and "order" not in f for f in others) == 22
    bj = json.loads(Path(pb).read_text())          # a bundle that DOES carry the lists
    for p, m in MAPS.items():
        ss = sorted(m)
        bj["phases"][str(p)] = {"seeds": ss, "match_rates": [m[s] for s in ss]}
    bj["bidirectional"] = {"constant": [1002, 1100], "variable": [1100, 1250]}
    bj["phase_counts"] = {str(p): {"survivor_rows": len(m), "distinct_survivors": len(m)}
                          for p, m in MAPS.items()}
    Path(pb).write_text(json.dumps(bj))
    rc2, rep2 = compare(pa, pb)
    st2 = status(rep2)
    for f in nulls:
        assert st2.get(f) == CEO.NOT_COMPARED, (f, st2.get(f))
    assert rc2 == 3, rc2
    return ("same run twice: exit 3 SAME WHERE COMPARED, the 10 nulls NOT COMPARED, all else "
            "EQUAL (22 arrays); against a bundle that carries the lists: still NOT COMPARED")


def g_mixed_source():
    """Beta Q6: the miner ledger of run A paired with run B's publication, same job, both
    trial 0, B differing by one seed (its variable bidirectional set loses 1250)."""
    maps_b = {k: dict(v) for k, v in MAPS.items()}
    del maps_b[4][1250]
    maps_b[4][1249] = 0.5                                  # same count, one seed moved
    run_a = build_run("mixed_A")
    run_b = build_run("mixed_B", maps=maps_b)
    base_a, staging_a, outroot_a, pj_a, _ = run_a
    rcpt_a = receipt(run_a)
    # (a) A's receipt, B's publication: refused, the bound generation differs
    mixed = (base_a, staging_a, run_b[2], pj_a, None)
    msg = refused(lambda: bundle(mixed, rcpt_a, name="mixed_bundle"), "does not match its receipt")
    assert "generation_id" in msg and "artifact_sha256" in msg, msg[:300]
    # (b) a receipt written for the mixed pair, A having finished well before B published:
    #     refused by B6 even though B1-B5 and B7-B8 all pass for the same job
    now = time.time()
    set_trial_times(staging_a, now - 1000 - TRIAL_SECONDS, now - 1000)
    refused(lambda: receipt(mixed, name="mixed_receipt"), "B6")
    # (c) the correctly paired arms still compare, and differ only where B was changed
    run_a2 = build_run("mixed_A2")
    pa, _ = extract(run_a2, name="A")
    pb, _ = extract(run_b, name="B")
    rc, rep = compare(pa, pb)
    diff = sorted(f for f, s in status(rep).items() if s == CEO.DIFFERENT)
    assert rc == 1 and diff and all(f.startswith("npz.") for f in diff), (rc, diff)
    return ("A's ledger + B's publication: refused against A's receipt (generation, artifact); "
            "a receipt for the mixed pair refused by B6; the true pairs: exit 1 at " + ", ".join(diff))


def forge_credit(run, mutate=lambda c, l: True):
    """Write a coverage row and a link row for the run's CURRENT generation by hand (the
    coverage table is append-only, so a forged row is an INSERT), after `mutate(cov, link)`
    has edited them; `mutate` returning False leaves the coverage row out."""
    db = os.path.join(run[2], "prng_analysis.db")
    prov = json.loads(Path(run[2], ".s172_accumulator", "current", "provenance.json").read_text())
    con = sqlite3.connect(db)
    con.execute(SSL._DDL[SSL.T_LINK])
    cov = {"coverage_id": "cov-1", "run_id": "step1_run", "study_identity": None,
           "prng_base": "java_lcg", "skip_modes_executed": "constant,variable",
           "seed_domain_contract": prov["seed_domain_contract"], "seed_start": S0,
           "seed_end_exclusive": S0 + N, "dataset_sha256": DSHA, "repository_commit": "0" * 40,
           "artifact_sha256": prov["artifact_sha256"], "generation_id": prov["generation_id"],
           "publication_status": "CERTIFIED", "recorded_at": "2026-11-21T00:00:00Z"}
    lnk = {"link_id": "l1", "coverage_id": "cov-1", "search_id": "srch1", "chunk_id": "c1",
           "prng_base": "java_lcg", "skip_modes_executed": '["constant","variable"]',
           "seed_start": S0, "seed_end_exclusive": S0 + N,
           "generation_id": prov["generation_id"], "trials_complete": 1, "trials_pruned": 0,
           "trials_fail": 0, "trials_other": 0, "recorded_utc": "2026-11-21T00:00:00Z"}
    if mutate(cov, lnk):
        con.execute(f"INSERT INTO certified_coverage ({','.join(cov)}) VALUES "
                    f"({','.join('?' * len(cov))})", tuple(cov.values()))
    con.execute(f"INSERT INTO {SSL.T_LINK} ({','.join(lnk)}) VALUES "
                f"({','.join('?' * len(lnk))})", tuple(lnk.values()))
    con.commit()
    con.close()
    _checkpoint(db)
    return run


def g_coverage_cross_id():
    """Beta Q3: coverage must name this generation; the link must name that coverage."""
    def forged(tag, mutate):
        return forge_credit(build_run(tag, certify=False, link=False), mutate)

    ok = forged("cov_ok", lambda c, l: True)
    _, b = extract(ok)
    assert b["credit"]["granted"] and b["provenance"]["coverage_id"] == "cov-1"
    cases = [
        ("cov_gen", lambda c, l: c.update(generation_id="gen-other") or True, "B7: the certified_coverage row"),
        ("cov_interval", lambda c, l: c.update(seed_end_exclusive=S0 + N + 1) or True, "B7: certified interval"),
        ("cov_dataset", lambda c, l: c.update(dataset_sha256="ef" * 32) or True, "B7: certified dataset_sha256"),
        ("link_cov", lambda c, l: l.update(coverage_id="cov-other") or True, "B8: link"),
        ("link_interval", lambda c, l: l.update(seed_end_exclusive=S0 + N - 1) or True, "B8: link"),
        ("link_no_cov", lambda c, l: False, "B8: a step1_chunk_link row names"),
    ]
    for tag, m, needle in cases:
        run = forged(tag, m)
        refused(lambda: extract(run), needle)
    return ("coverage naming another generation, interval or dataset: refused (B7); link naming "
            "another coverage, interval, or no coverage: refused (B8)")


def g_receipt():
    run = build_run("receipt")
    base, staging, outroot, pj, _ = run
    refused(lambda: bundle(run, None), "--receipt is required")
    r = receipt(run)
    refused(lambda: receipt(run), "already exists")
    refused(lambda: EXT.write_receipt(staging, outroot, pj, os.path.join(outroot, "r.json"),
                                      live_paths=NO_LIVE), "inside the run")
    tam = json.loads(Path(r).read_text())
    tam["miner_run_id"] = "cfg_t0_other"
    t2 = os.path.join(base, "tampered.json")
    Path(t2).write_text(json.dumps(tam))
    refused(lambda: bundle(run, t2), "does not match its receipt")
    _, b = bundle(run, r)
    assert b["run_id"] == RUN_ID
    return "no receipt, an existing receipt, a receipt inside the run, a changed bound field: refused"


def g_f1_warning():
    run = build_run("f1")
    _, b = extract(run)
    sc = b["provenance"]["constant_skip_range_code_reading"]
    assert sc["effective"] == [0, 16], sc
    assert any("F-1" in c and "[6, 99]" in c for c in b["provenance"]["self_checks"]), b["provenance"]["self_checks"]
    fake = os.path.join(run[0], "fake_tree", "miner")
    os.makedirs(fake)
    shutil.copy(os.path.join(ROOT, "miner", "range_miner_worker.py"), fake)
    Path(fake, "range_miner_coordinator.py").write_text(
        "def build_stripe_assign_payload(x):\n    return {'skip_range': x}\n")
    _, b2 = extract(run, name="b2", code_root=os.path.dirname(fake))
    assert b2["provenance"]["constant_skip_range_code_reading"]["effective"] is None
    assert b2["provenance"]["job_identity"]["constant_skip_range_effective"] is None
    assert any("not known" in c for c in b2["provenance"]["self_checks"])
    return "F-1 warning: code reading [0, 16] vs requested [6, 99]; a payload naming skip_range: 'not known'"


def g_credit():
    run = build_run("nocov", certify=False)
    _, b = extract(run)
    assert b["credit"]["granted"] is False and b["credit"]["certified_interval"] is None, b["credit"]
    assert b["inputs"]["run.dataset_sha256"] is None, b["inputs"]["run.dataset_sha256"]
    run2 = build_run("nolink", link=False)
    _, b2 = extract(run2)
    assert b2["credit"]["granted"] is False and b2["credit"]["linked_interval"] is None
    assert b2["credit"]["certified_interval"] == [S0, S0 + N]
    run3 = build_run("noledger")
    os.remove(os.path.join(run3[2], "prng_analysis.db"))
    refused(lambda: extract(run3), "not found")
    return ("no coverage row: granted false, dataset_sha256 null; no link table: granted false "
            "(the agreed definition needs the link), linked null; "
            "no run ledger: refused")


def g_params():
    run = build_run("params_pwc", params={"use_range_miner": False, "use_persistent_workers": True})
    refused(lambda: extract(run), "not a RANGE-MINER run")
    run2 = build_run("params_absent", params={"use_range_miner": DROP})
    refused(lambda: extract(run2), "not a RANGE-MINER run")
    run3 = build_run("params_nostage", params={"staging_dir": DROP})
    refused(lambda: extract(run3), "B1")
    run4 = build_run("params_otherstage", params={"staging_dir": "/home/michael/elsewhere"})
    refused(lambda: extract(run4), "B1")
    return "PWC params, no explicit use_range_miner, no staging_dir, another staging_dir: refused"


def g_pins():
    for pin, val in (("warm_start_window", 9), ("warm_start_fwd_thresh", 0.72),
                     ("warm_start_skip_max", 16)):
        run = build_run(f"pin_{pin}", params={pin: val})
        refused(lambda: extract(run), "B2")
    run = build_run("pins_absent", params={k: DROP for k in PINS})
    extract(run)                                     # no pins: nothing to bind, not refused
    return "a window, threshold or skip pin that differs from trial_context: refused (B2)"


def g_live_path():
    run = build_run("live")
    refused(lambda: extract(run, live_paths=(run[1],)), "live path")
    refused(lambda: extract(run, name="b2", live_paths=(run[2],)), "live path")
    return "staging dir or output root under a live path: refused"


def g_missing_generation():
    base, staging, outroot, pj, _ = build_run("nogen")
    shutil.rmtree(os.path.join(outroot, ".s172_accumulator"))
    refused(lambda: extract((base, staging, outroot, pj, None)), "no published generation")
    assert not os.path.exists(os.path.join(base, "bundle", "bundle.json"))
    return "no published generation: refused, no bundle"


def g_null_count():
    run = build_run("nullcount", mutate=lambda c: c.execute(
        # p3_s1's shards sum to 0, so a NULL read as 0 would pass the shard check
        "UPDATE stripes SET survivors_total=NULL WHERE stripe_id='p3_s1'"))
    _, b = extract(run)
    assert b["phase_counts"]["3"]["survivor_rows"] is None, b["phase_counts"]["3"]
    assert b["phase_counts"]["1"]["survivor_rows"] == len(MAPS[1])
    return "a NULL survivors_total: phase 3 survivor_rows null (NOT COMPARED), never 0"


def g_phase_gap():
    run = build_run("gap", mutate=lambda c: c.execute(
        # an interior gap: the stripes still end at the generation's end, so only the
        # contiguity rule (not B5) can see it
        "UPDATE stripes SET seed_start=seed_start+1, seed_count=seed_count-1 "
        "WHERE stripe_id='p2_s1'"))
    refused(lambda: extract(run), "gap or overlap")
    return "a gap between phase-2 stripes: refused"


def g_non_done():
    run = build_run("nondone", mutate=lambda c: c.execute(
        "UPDATE stripes SET state='staging' WHERE stripe_id='p4_s1'"))
    refused(lambda: extract(run), "a stripe is not done")
    return "a committed trial with a phase-4 stripe not done: refused"


def g_phase_set():
    # an extra phase first: its stripes sweep the generation's interval and B4 has no NPZ
    # rows for it, so only the phase-set check can refuse it
    run2 = build_run("phaseset_extra", mutate=lambda c: c.execute(
        "INSERT INTO stripes (run_id, stripe_id, seed_start, seed_count, state, current_attempt, "
        "phase, family_name, survivors_total, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (RUN_ID, "p5_s0", S0, N, "done", 0, 5, "java_lcg", 0, 0.0)))
    refused(lambda: extract(run2), "stripe phases [1, 2, 3, 4, 5] != the phases")
    run = build_run("phaseset_missing", phases=(1, 2))      # generation says both modes
    refused(lambda: extract(run), "stripe phases [1, 2] != the phases")
    return "stripe phases {1..5} or {1,2} against a both-modes generation: refused"


def g_trial_ordinal():
    run = build_run("ordinal", ctx=dict(CTX, trial_number=1))
    refused(lambda: extract(run), "B3: the NPZ carries trial_number(s) [1]")
    return "NPZ rows of trial 1 against ledger trial 0: refused (B3)"


def g_npz_context():
    run = build_run("npzctx", ctx=dict(CTX, window_size=9))
    refused(lambda: extract(run), "B3: NPZ window_size")
    run2 = build_run("npzctx_skip", ctx=dict(CTX, skip_min=0, skip_max=16))
    refused(lambda: extract(run2), "B3: NPZ skip_min")
    return "NPZ rows with another window_size or skip range than the trial: refused (B3)"


def g_npz_counts():
    maps = {k: dict(v) for k, v in MAPS.items()}
    maps[1][1350] = 0.5                                  # NPZ forward_count 5, ledger phase 1: 4
    run = build_run("npzcounts", maps=maps)
    refused(lambda: extract(run), "B4: NPZ constant rows carry forward_count [5]")
    return "NPZ forward_count 5 against the trial's phase-1 count 4: refused (B4)"


def g_swept_interval():
    run = build_run("swept", seed_count=N + 100)
    refused(lambda: extract(run), "B5")
    return "stripes swept [1000, 1400) but the generation declares [1000, 1500): refused (B5)"


def g_lag():
    run = build_run("lag")
    now = time.time()
    set_trial_times(run[1], now - 1000 - TRIAL_SECONDS, now - 1000)
    refused(lambda: extract(run), "B6")
    run2 = build_run("lag_neg")
    set_trial_times(run2[1], now + 100 - TRIAL_SECONDS, now + 100)
    refused(lambda: extract(run2), "B6")
    return "published 1000 s after a 60 s trial, or before its commit: refused (B6)"


def g_uncommitted():
    run = build_run("running", state="running")
    refused(lambda: extract(run), "only a committed")
    return "an uncommitted trial: refused"


def _rewrite_npz(res, arrays, order, fix_sidecar):
    for p in (res.binary_npz_path, res.all_npz_path):
        if os.path.exists(p):
            os.remove(p)
    np.savez_compressed(str(res.binary_npz_path), **{n: arrays[n] for n in order})
    shutil.copyfile(res.binary_npz_path, res.all_npz_path)
    if fix_sidecar:
        side = json.loads(Path(res.sidecar_path).read_text())
        side["artifact_sha256"] = sha(res.binary_npz_path)
        Path(res.sidecar_path).write_text(json.dumps(side))


def g_truncated():
    run = build_run("trunc", certify=False, link=False)
    res = run[4]
    data = Path(res.binary_npz_path).read_bytes()
    os.remove(res.binary_npz_path)
    Path(res.binary_npz_path).write_bytes(data[: len(data) // 2])
    refused(lambda: extract(run), "truncated or modified")
    side = json.loads(Path(res.sidecar_path).read_text())          # sidecar made to agree
    side["artifact_sha256"] = hashlib.sha256(data[: len(data) // 2]).hexdigest()
    Path(res.sidecar_path).write_text(json.dumps(side))
    refused(lambda: extract(run, name="b2"), "cannot read")
    return "a half-length NPZ: refused on sha256, and refused as unreadable when the sidecar agrees"


def g_modified():
    run = build_run("modified", certify=False, link=False)   # the sha check is the only guard
    res = run[4]
    with np.load(res.binary_npz_path) as d:
        arrs = {n: d[n].copy() for n in d.files}
    arrs["forward_matches"][0] = np.float32(0.1)
    _rewrite_npz(res, arrs, list(CEO.ARRAY_NAMES), fix_sidecar=False)
    refused(lambda: extract(run), "truncated or modified")
    return "a readable NPZ with one changed value, sidecar not updated: refused"


def g_wrong_order():
    good_run = build_run("order_good", certify=False, link=False)
    forge_credit(good_run)
    good, _ = extract(good_run, name="good")
    run = build_run("order", certify=False, link=False)
    res = run[4]
    with np.load(res.binary_npz_path) as d:
        arrs = {n: d[n].copy() for n in d.files}
    order = list(CEO.ARRAY_NAMES)
    i, j = order.index("forward_matches"), order.index("reverse_matches")
    order[i], order[j] = order[j], order[i]
    _rewrite_npz(res, arrs, order, fix_sidecar=True)
    forge_credit(run)                                   # certify the rewritten artifact
    bad, b = extract(run, name="bad")
    bad_npz = os.path.join(os.path.dirname(bad), "survivors.npz")
    assert Path(bad_npz).read_bytes() == Path(res.binary_npz_path).read_bytes()
    with zipfile.ZipFile(bad_npz) as z:
        assert [n[:-4] for n in z.namelist()] == order, "the extractor re-ordered the NPZ"
    assert b["provenance"]["npz_array_order_matches_contract"] is False
    rc, rep = compare(good, bad)
    st = status(rep)
    assert st["npz.order[B]"] == CEO.DIFFERENT and st["npz.order[A]"] == CEO.EQUAL and rc == 1, (rc, st)
    return "swapped array order: copied as is, flagged, the comparator reports npz.order[B] (exit 1)"


def g_retry():
    def m(c):
        c.execute("UPDATE stripes SET current_attempt=1 WHERE stripe_id='p1_s0'")
        for sub in range(2):        # attempt 1 is current; attempt 0's rows stay (superseded)
            c.execute("INSERT INTO shards (run_id, stripe_id, attempt, sub_index, worker_id, "
                      "seed_start, seed_count, survivor_count, staging_status, created_at, "
                      "local_cleanup_status) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                      (RUN_ID, "p1_s0", 1, sub, "rig9:gpu0", S0 + sub * 100, 100,
                       len(MAPS[1]) if sub == 0 else 0, "verified", 101.0, "deleted"))
    run = build_run("retry", mutate=m)
    _, b = extract(run)
    assert b["phase_counts"]["1"]["survivor_rows"] == len(MAPS[1]), b["phase_counts"]["1"]
    return "a retried stripe: counted once, from its current attempt"


def g_wal():
    run = build_run("wal", state="running")
    staging, outroot = run[1], run[2]
    db = os.path.join(staging, "miner_ledger.db")
    w = sqlite3.connect(db)
    w.execute("PRAGMA journal_mode=WAL")
    w.execute("PRAGMA wal_autocheckpoint=0")
    w.execute("UPDATE trials SET state='committed', commit_delivery_status='done'")
    w.commit()                                 # committed only in the -wal, connection held
    try:
        assert os.path.getsize(db + "-wal") > 0
        before = snapshot(staging, outroot)
        _, b = extract(run)
        assert snapshot(staging, outroot) == before, "the extractor wrote beside the ledger"
        assert b["run_id"] == RUN_ID
    finally:
        w.close()
    return "non-empty -wal: read from a byte copy (sees the WAL-only commit); nothing changed beside the ledger"


def g_prior():
    run = build_run("prior", prior=True)
    refused(lambda: extract(run), "has a parent")
    return "a generation with a parent: refused"


def g_multi_trial():
    run = build_run("multi", extra_trials=(("cfg_t1_feedface", 1, "committed"),))
    refused(lambda: extract(run), "name one")
    _, b = extract(run, name="b2", run_id=RUN_ID)
    assert b["run_id"] == RUN_ID
    return "two trials in the ledger: refused without --run-id"


def g_out_dir():
    base, staging, outroot, pj, _ = run = build_run("outdir")
    r = receipt(run)
    refused(lambda: EXT.build_bundle(staging, outroot, pj, os.path.join(staging, "x"), r,
                                     live_paths=NO_LIVE), "inside the run")
    refused(lambda: EXT.build_bundle(staging, outroot, pj, os.path.join(outroot, "x"), r,
                                     live_paths=NO_LIVE), "inside the run")
    bundle(run, r, name="used")
    refused(lambda: bundle(run, r, name="used"), "not empty")
    return "--out-dir inside the run, or not empty: refused"


def g_cli():
    base, staging, outroot, pj, _ = build_run("cli")
    rc_path, out = os.path.join(base, "cli_receipt.json"), os.path.join(base, "cli")
    common = [sys.executable, TOOL, "--staging-dir", staging, "--output-root", outroot,
              "--params-json", pj]
    r = subprocess.run(common + ["--write-receipt", rc_path], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "RECEIPT_WRITTEN" in r.stdout, (r.returncode, r.stderr[-400:])
    r = subprocess.run(common + ["--out-dir", out], capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "--receipt is required" in r.stderr, (r.returncode, r.stderr[-400:])
    cmd = common + ["--out-dir", out, "--receipt", rc_path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "null (NOT COMPARED): phases.1" in r.stdout, (r.returncode, r.stderr[-400:])
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "not empty" in r.stderr, (r.returncode, r.stderr[-400:])
    return "as a program: receipt exit 0; bundle without receipt exit 2; bundle exit 0; no overwrite exit 2"


GATES = [
    ("G-COMPARATOR", g_comparator), ("G-CLEAN", g_clean), ("G-NULL-NEVER-EQUAL", g_null_never_equal),
    ("G-MIXED-SOURCE", g_mixed_source), ("G-COVERAGE-CROSS-ID", g_coverage_cross_id),
    ("G-RECEIPT", g_receipt), ("G-F1-WARNING", g_f1_warning), ("G-CREDIT", g_credit),
    ("G-PARAMS", g_params), ("G-PINS", g_pins), ("G-LIVE-PATH", g_live_path),
    ("G-MISSING-GENERATION", g_missing_generation), ("G-NULL-COUNT", g_null_count),
    ("G-PHASE-GAP", g_phase_gap), ("G-NON-DONE", g_non_done), ("G-PHASE-SET", g_phase_set),
    ("G-TRIAL-ORDINAL", g_trial_ordinal), ("G-NPZ-CONTEXT", g_npz_context),
    ("G-NPZ-COUNTS", g_npz_counts), ("G-SWEPT-INTERVAL", g_swept_interval), ("G-LAG", g_lag),
    ("G-UNCOMMITTED", g_uncommitted), ("G-TRUNCATED-NPZ", g_truncated),
    ("G-MODIFIED-NPZ", g_modified), ("G-WRONG-ORDER", g_wrong_order), ("G-RETRY", g_retry),
    ("G-WAL", g_wal), ("G-PRIOR-GENERATION", g_prior), ("G-MULTI-TRIAL", g_multi_trial),
    ("G-OUT-DIR", g_out_dir), ("G-CLI", g_cli),
]

# one source change each, and the gate that must catch it
MUTANTS = [
    ("MU1 a missing phase written as an empty list",
     "        phases[ph] = None\n", '        phases[ph] = {"seeds": [], "match_rates": []}\n',
     g_null_never_equal),
    ("MU2 a missing bidirectional set written as an empty list",
     "        bidi[m] = None\n", "        bidi[m] = []\n", g_null_never_equal),
    ("MU3 NPZ sha256 not checked against the sidecar",
     'if sha != prov.get("artifact_sha256"):', "if False:", g_modified),
    ("MU4 NPZ re-saved in contract order instead of copied",
     'shutil.copyfile(ev["_npz_path"], out_npz)',
     'np.savez(out_npz[:-4], **{n: np.load(ev["_npz_path"])[n] for n in CEO.ARRAY_NAMES})',
     g_wrong_order),
    ("MU5 swept interval ignores contiguity", "            if s != cur:", "            if False:",
     g_phase_gap),
    ("MU6 counts sum shards of every attempt",
     "sh[0] == s[0] and sh[1] == s[5]", "sh[0] == s[0]", g_retry),
    ("MU7 prior generation accepted",
     'if prov.get("parent_generation_id") is not None or prov.get("prior_row_count") != 0:',
     "if False:", g_prior),
    ("MU8 immutable read even with a non-empty -wal",
     "if not (os.path.exists(wp) and os.path.getsize(wp) > 0):", "if True:", g_wal),
    ("MU9 an uncommitted trial accepted",
     'if t[2] != "committed" or t[3] != "done":', "if False:", g_uncommitted),
    ("MU10 a NULL count read as zero",
     "bad = [s[0] for s in st if s[6] is None]",
     "bad = []\n        st = [s[:6] + (s[6] or 0,) + s[7:] for s in st]", g_null_count),
    ("MU11 distinct_survivors claimed equal to the row count",
     '{"survivor_rows": rows, "distinct_survivors": None}',
     '{"survivor_rows": rows, "distinct_survivors": rows}', g_clean),
    ("MU12 dataset_sha256 taken from trial_context, not certified_coverage",
     'CEO.build_inputs(ev["params"], prov, cov[2] if cov else None)',
     'CEO.build_inputs(ev["params"], prov, led["ctx"]["dataset_sha256"])', g_credit),
    ("MU13 params engine check dropped",
     'if params.get("use_range_miner") is not True or params.get("use_persistent_workers") is True:',
     "if False:", g_params),
    ("MU14 live-path refusal dropped",
     "        refuse_live(p, what, live_paths)\n", "        pass\n", g_live_path),
    ("MU15 B1 params staging_dir not bound",
     "if not isinstance(p_stage, str) or _real(p_stage) != staging_dir:", "if False:", g_params),
    ("MU16 B2 pins not bound", "    if bad:\n        raise ExtractRefused(f\"B2",
     "    if False:\n        raise ExtractRefused(f\"B2", g_pins),
    ("MU17 B3 trial ordinal not checked", "if tns and tns != [int(t[1])]:", "if False:",
     g_trial_ordinal),
    ("MU18 B3 window/skip not checked", "if vals and vals != [int(ctx[col])]:", "if False:",
     g_npz_context),
    ("MU19 B4 counts not checked", "if vals and mine is not None and vals != [mine]:",
     "if False:", g_npz_counts),
    ("MU20 B5 swept interval only a warning",
     '        raise ExtractRefused(f"B5: swept intervals {off} != the generation\'s {req}")',
     "        pass", g_swept_interval),
    ("MU21 B6 publication lag not checked", "if not 0 <= lag <= duration:", "if False:", g_lag),
    ("MU22 B7 coverage generation_id ignored",
     'if cov[4] != prov.get("generation_id"):', "if False:", g_coverage_cross_id),
    ("MU23 B8 link coverage_id/interval ignored",
     "if link[3] != cov[3] or [int(link[0]), int(link[1])] != [int(cov[0]), int(cov[1])]:",
     "if False:", g_coverage_cross_id),
    ("MU24 receipt not compared", "        if diff:\n", "        if False:\n", g_mixed_source),
    ("MU25 non-done stripes accepted", 'if any(st != "done" for _, _, st in items):',
     "if False:", g_non_done),
    ("MU26 phase set not checked", "if sorted(by_phase) != sorted(required):", "if False:",
     g_phase_set),
]


def run_mutants():
    global EXT
    results = []
    for name, old, new, gate in MUTANTS:
        assert TOOL_SRC.count(old) == 1, f"{name}: anchor found {TOOL_SRC.count(old)} times"
        EXT = load_tool(TOOL_SRC.replace(old, new))
        why = ""
        try:
            gate()
            caught = False
        except Exception as exc:                                      # noqa: BLE001
            caught, why = True, f"{type(exc).__name__}: {str(exc)[:160]}"
        finally:
            EXT = load_tool()
        results.append((name, gate.__name__, caught, why))
    return results


def main():
    global EXT
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    print("VERIFICATION test_extract_miner_bundle START", flush=True)
    print(f"comparator: {CMP} sha256 {sha(CMP)}", flush=True)
    EXT = load_tool()
    passed = 0
    for name, fn in GATES:
        try:
            print(f"[PASS] {name}: {fn()}", flush=True)
            passed += 1
        except Exception as exc:                                     # noqa: BLE001
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}", flush=True)
    shutil.rmtree(OUT)
    os.makedirs(OUT)
    killed = 0
    for name, gate, caught, why in run_mutants():
        print(f"[{'KILLED' if caught else 'SURVIVED'}] {name} (by {gate}) {why}", flush=True)
        killed += caught
    total = len(GATES) + len(MUTANTS)
    print(f"\n{passed}/{len(GATES)} gates PASS; {killed}/{len(MUTANTS)} mutants KILLED")
    print("RESULT: PASS" if passed + killed == total else "RESULT: FAIL")
    print("VERIFICATION test_extract_miner_bundle COMPLETE", flush=True)
    return 0 if passed + killed == total else 1


if __name__ == "__main__":
    sys.exit(main())
