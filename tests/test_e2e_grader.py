#!/usr/bin/env python3
"""
Self-test for scripts/diagnostics/e2e_grader.py (WP-3 first form). CPU only, small fixtures.

Run:  python3 tests/test_e2e_grader.py            (all gates)
      python3 tests/test_e2e_grader.py --only NAME (one gate; used by the mutation runner)

Every assertion carries a tag "[A-...]" so the mutation runner can prove that a mutant was
killed by the assertion credited to it, not by an unrelated crash.
Fixtures and grader outputs are written under <repo>/data/e2e_grader_selftest/ (never /tmp,
CLAUDE.md box rule 7). The grader under test is E2E_GRADER_PATH (default: the real file).
Seeds are vehicles; no test here judges a seed. Fixtures are synthetic draws.
"""
import contextlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import date, timedelta
from itertools import product
from pathlib import Path

import numpy as np
from scipy.stats import binom

REPO = Path(__file__).resolve().parents[1]
GRADER_PATH = Path(os.environ.get("E2E_GRADER_PATH",
                                  REPO / "scripts" / "diagnostics" / "e2e_grader.py"))
WORK = Path(os.environ.get("E2E_SELFTEST_DIR", REPO / "data" / "e2e_grader_selftest" /
                           time.strftime("run_%Y%m%dT%H%M%S")))


def _load_grader():
    spec = importlib.util.spec_from_file_location("e2e_grader_under_test", GRADER_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod      # so the control's worker pool can pickle its functions
    spec.loader.exec_module(mod)
    return mod


G = _load_grader()
TESTS = []


def gate(fn):
    TESTS.append(fn)
    return fn


def check(cond, tag, msg=""):
    if not cond:
        raise AssertionError(f"[{tag}] {msg}")


# ------------------------------------------------------------------------- fixtures ------
def make_dataset(n=400, seed=7, zero_at=(203,)):
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, 1000, n).tolist()
    for z in zero_at:
        if z < n:
            draws[z] = 0
    d0 = date(2000, 1, 1)
    rows = []
    for i in range(n):
        rows.append({"date": (d0 + timedelta(days=i // 2)).isoformat(),
                     "session": "midday" if i % 2 == 0 else "evening", "draw": int(draws[i])})
    return rows


def excl(y, k):
    return [(y + 1 + j) % 1000 for j in range(k)]


def incl(y, k):
    return [y] + [(y + 1 + j) % 1000 for j in range(k - 1)]


class Case:
    """One grader invocation on files written under WORK/<name>/."""

    def __init__(self, name, dataset=None):
        self.dir = WORK / name
        if self.dir.exists():
            shutil.rmtree(self.dir)
        self.dir.mkdir(parents=True)
        self.dataset = dataset if dataset is not None else make_dataset()
        self.ds_path = self.dir / "dataset.json"
        self.ds_path.write_text(json.dumps(self.dataset))
        self.ds_sha = G.sha256_file(self.ds_path)
        self.gb = {"schema": G.SCHEMA_GRADE_BLOCK, "name": name, "start_index": 200,
                   "end_index": len(self.dataset), "sessions": ["evening", "midday"],
                   "frequency_control_block": {"start_index": 0, "end_index": 200}}
        self.arms = {"schema": G.SCHEMA_ARMS, "arms": [
            {"arm_id": "A0", "config_id": "cfgA0", "upstream_unseen_from_index": 200}]}
        self.abst = {"schema": G.SCHEMA_ABSTENTION, "allow_abstention": False,
                     "allowed_reasons": None, "empty_pool": "error"}
        self.records = []
        self.raw_lines = None

    def y(self, t):
        return self.dataset[t]["draw"]

    def rec(self, arm, t, pool, cfg=None, cutoff=None, **kw):
        if t < len(self.dataset):
            tgt = {"combined_index": t, "date": self.dataset[t]["date"],
                   "session": self.dataset[t]["session"]}
        else:
            tgt = {"combined_index": t, "date": "2099-01-01", "session": "midday"}
        r = {"schema": G.SCHEMA_POOL, "arm_id": arm, "config_id": cfg or f"cfg{arm}",
             "dataset_sha256": self.ds_sha, "data_cutoff_index": t if cutoff is None else cutoff,
             "target": tgt, "pool": pool}
        r.update(kw)
        return r

    def fill(self, arm, fn, start=200, end=None):
        for t in range(start, end if end is not None else len(self.dataset)):
            self.records.append(self.rec(arm, t, fn(t, self.y(t))))

    def run(self, *extra, reps=20, workers=1, provenance=False, out_name="out"):
        for nm, obj in (("grade_block.json", self.gb), ("arms.json", self.arms),
                        ("abstention.json", self.abst)):
            (self.dir / nm).write_text(json.dumps(obj))
        pools = self.dir / "pools.jsonl"
        if self.raw_lines is not None:
            pools.write_text(self.raw_lines)
        else:
            pools.write_text("".join(json.dumps(r) + "\n" for r in self.records))
        self.out = self.dir / out_name
        argv = ["--pools", str(pools), "--dataset", str(self.ds_path),
                "--dataset-sha256", self.ds_sha, "--grade-block", str(self.dir / "grade_block.json"),
                "--arms", str(self.dir / "arms.json"), "--abstention", str(self.dir / "abstention.json"),
                "--out-dir", str(self.out), "--control-reps", str(reps), "--workers", str(workers)]
        if not provenance:
            argv.append("--no-provenance")
        argv += list(extra)
        self.argv = argv
        err, outbuf = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(outbuf):
                rc = G.main(argv)
        except Exception as e:  # a crash is not the fail-closed error path
            rc = f"EXC:{type(e).__name__}:{e}"
        self.stderr, self.stdout = err.getvalue(), outbuf.getvalue()
        self.summary = (json.loads((self.out / "summary.json").read_text())
                        if (self.out / "summary.json").is_file() else None)
        self.rows = ([json.loads(l) for l in (self.out / "per_draw.jsonl").read_text().splitlines()]
                     if (self.out / "per_draw.jsonl").is_file() else None)
        return rc

    def result(self, arm):
        return next(r for r in self.summary["results"] if r["arm_id"] == arm)

    def row(self, arm, t):
        return next(r for r in self.rows if r["arm_id"] == arm and r["combined_index"] == t)


# ---------------------------------------------------------------------------- gates ------
@gate
def test_known_count():
    """Known hit count on varied k, duplicates of the winning value included."""
    c = Case("known_count")
    ks = [1, 5, 20, 37]

    def fn(t, y):
        k = ks[t % 4]
        if t % 7 == 0:
            p = incl(y, k)
            return p + [y] if t % 2 == 0 else p
        return excl(y, k)
    c.fill("A0", fn)
    rc = c.run()
    check(rc == 0, "A-RC0", f"rc={rc} {c.stderr}")
    want_hits = sum(1 for t in range(200, 400) if t % 7 == 0)
    want_exp = sum(ks[t % 4] for t in range(200, 400)) / 1000
    s = c.result("A0")["stats"]
    check(s["hits"] == want_hits, "A-HITCOUNT", f"{s['hits']} != {want_hits}")
    check(abs(s["expected_hits"] - want_exp) < 1e-12, "A-EXPECTED", f"{s['expected_hits']} != {want_exp}")
    check(s["n_graded"] == 200, "A-NGRADED", str(s["n_graded"]))
    check(len(c.rows) == 200, "A-ROWS", str(len(c.rows)))
    check(c.summary["outputs"]["per_draw.jsonl"]["sha256"] == G.sha256_file(c.out / "per_draw.jsonl"),
          "A-OUTSHA", "per_draw sha mismatch")
    sums = (c.out / "SHA256SUMS").read_text().split()
    check(sums[sums.index("summary.json") - 1] == G.sha256_file(c.out / "summary.json"),
          "A-SUMSHA", "SHA256SUMS summary mismatch")
    r = c.row("A0", 210)  # 210 % 7 == 0 and even: duplicate winning value
    check(r["hit"] == 1 and r["n_duplicate_values"] == 1 and r["k"] == ks[210 % 4],
          "A-DUP-ONEHIT", str(r))
    kc = c.summary["controls"]["A0"]["known_count"]
    check(kc["passed"] and kc["expected_hits"] == 100, "A-KC-RUNTIME", str(kc))
    check(c.summary["inputs"]["dataset"]["sha256"] == c.ds_sha, "A-DSHA-REC", "dataset sha not recorded")


@gate
def test_dup_values():
    c = Case("dup_values")
    c.fill("A0", lambda t, y: excl(y, 3))
    y = c.y(220)
    c.records[20] = c.rec("A0", 220, [y, y, 5, 5, 6])
    rc = c.run()
    check(rc == 0, "A-RC0", f"rc={rc} {c.stderr}")
    r = c.row("A0", 220)
    distinct = len({y, 5, 6})
    check(r["k"] == distinct and r["k_raw"] == 5, "A-K-DISTINCT", str(r))
    check(r["n_duplicate_values"] == 5 - distinct, "A-K-DISTINCT", str(r))
    check(r["hit"] == 1, "A-DUP-ONEHIT", str(r))
    check(r["pool_sha256"] == G.pool_sha256([y, y, 5, 5, 6]), "A-POOLSHA", "pool sha of raw pool")


@gate
def test_draw_000():
    c = Case("draw_000")
    c.fill("A0", lambda t, y: excl(y, 4))
    check(c.y(203) == 0, "A-FIXTURE", "fixture must have draw 000 at 203")
    c.records[3] = c.rec("A0", 203, [0, 11, 12])
    rc = c.run()
    check(rc == 0, "A-RC0", f"rc={rc} {c.stderr}")
    check(c.row("A0", 203)["hit"] == 1, "A-ZERO-HIT", str(c.row("A0", 203)))
    check(c.row("A0", 203)["outcome"] == 0, "A-ZERO-OUTCOME", "")


@gate
def test_out_of_range():
    for bad, tag in ((1000, "A-RANGE-HI"), (-1, "A-RANGE-LO")):
        c = Case(f"range_{tag}")
        c.fill("A0", lambda t, y: excl(y, 2))
        c.records[5] = c.rec("A0", 205, [1, bad])
        rc = c.run()
        check(rc == 2 and "outside [0, 999]" in c.stderr, tag, f"rc={rc} {c.stderr}")
        check(not c.out.exists(), tag, "nothing may be written on error")


@gate
def test_wrong_type():
    for bad, tag in ((True, "A-TYPE-BOOL"), (3.0, "A-TYPE-FLOAT"), ("12", "A-TYPE-STR"),
                     (None, "A-TYPE-NONE")):
        c = Case(f"type_{tag}")
        c.fill("A0", lambda t, y: excl(y, 2))
        c.records[5] = c.rec("A0", 205, [7, bad])
        rc = c.run()
        check(rc == 2 and "wrong type" in c.stderr, tag, f"rc={rc} {c.stderr}")
    c = Case("type_pool_not_list")
    c.fill("A0", lambda t, y: excl(y, 2))
    c.records[5] = c.rec("A0", 205, {"20": [1, 2]})
    rc = c.run()
    check(rc == 2 and "must be a list" in c.stderr, "A-TYPE-DICTPOOL", f"rc={rc} {c.stderr}")


@gate
def test_pending():
    base = Case("pending_base")
    base.fill("A0", lambda t, y: incl(y, 3) if t % 5 == 0 else excl(y, 3))
    rc0 = base.run()
    c = Case("pending")
    c.gb["end_index"] = 410
    c.fill("A0", lambda t, y: incl(y, 3) if t % 5 == 0 else excl(y, 3))
    c.records.append(c.rec("A0", 400, [1, 2, 3]))
    rc = c.run()
    check(rc0 == 0 and rc == 0, "A-RC0", f"rc={rc0},{rc} {c.stderr}")
    r = c.result("A0")
    check(r["n_pending"] == 1 and c.row("A0", 400)["status"] == "pending", "A-PENDING", str(r["n_pending"]))
    check(r["stats"]["n_graded"] == base.result("A0")["stats"]["n_graded"] and
          r["stats"]["hits"] == base.result("A0")["stats"]["hits"] and
          r["stats"]["expected_hits"] == base.result("A0")["stats"]["expected_hits"],
          "A-PENDING-NOTMISS", "pending changed the graded denominator or expectation")
    check(c.row("A0", 400)["hit"] is None, "A-PENDING-NOTMISS", "pending must carry no hit value")


@gate
def test_stale_target():
    c = Case("stale_date")
    c.fill("A0", lambda t, y: excl(y, 2))
    c.records[9]["target"]["date"] = "1999-12-31"
    rc = c.run()
    check(rc == 2 and "stale target" in c.stderr, "A-STALE", f"rc={rc} {c.stderr}")
    c = Case("stale_session")
    c.fill("A0", lambda t, y: excl(y, 2))
    c.records[9]["target"]["session"] = "evening" if c.records[9]["target"]["session"] == "midday" else "midday"
    rc = c.run()
    check(rc == 2 and "stale target" in c.stderr, "A-STALE", f"rc={rc} {c.stderr}")
    c = Case("stale_sha")
    c.fill("A0", lambda t, y: excl(y, 2))
    c.records[9]["dataset_sha256"] = "0" * 64
    rc = c.run()
    check(rc == 2 and "stale record" in c.stderr, "A-STALE-SHA", f"rc={rc} {c.stderr}")
    c = Case("stale_cfg")
    c.fill("A0", lambda t, y: excl(y, 2))
    c.records[9]["config_id"] = "cfgOTHER"
    rc = c.run()
    check(rc == 2 and "config_id" in c.stderr, "A-STALE-CFG", f"rc={rc} {c.stderr}")
    c = Case("stale_poolsha")
    c.fill("A0", lambda t, y: excl(y, 2))
    c.records[9]["pool_sha256"] = "f" * 64
    rc = c.run()
    check(rc == 2 and "pool_sha256" in c.stderr, "A-POOLSHA-BIND", f"rc={rc} {c.stderr}")


@gate
def test_duplicate_record():
    c = Case("dup_record_err")
    c.fill("A0", lambda t, y: excl(y, 2))
    c.records.append(c.rec("A0", 250, incl(c.y(250), 2)))
    rc = c.run()
    check(rc == 2 and "declared duplicate rule: error" in c.stderr, "A-DUP-RECORD", f"rc={rc} {c.stderr}")
    c = Case("dup_record_first")
    c.fill("A0", lambda t, y: excl(y, 2))
    c.records.append(c.rec("A0", 250, incl(c.y(250), 2)))
    rc = c.run("--duplicate-rule", "first_in_file")
    check(rc == 0, "A-RC0", f"rc={rc} {c.stderr}")
    dr = c.result("A0")["duplicates_resolved"]
    check(len(dr) == 1 and dr[0]["chosen_line"] == 51 and c.row("A0", 250)["hit"] == 0,
          "A-DUP-FIRST", str(dr))
    c = Case("dup_record_maxcut")
    c.fill("A0", lambda t, y: excl(y, 2))
    c.records[50]["data_cutoff_index"] = 240
    c.records.append(c.rec("A0", 250, incl(c.y(250), 2), cutoff=249))
    rc = c.run("--duplicate-rule", "max_data_cutoff")
    check(rc == 0 and c.row("A0", 250)["hit"] == 1 and c.row("A0", 250)["data_cutoff_index"] == 249,
          "A-DUP-MAXCUT", f"rc={rc} {c.stderr}")


@gate
def test_repeated_draw():
    """R26 (Beta X-E2E-GRADER-LANDED R1 Q2): row 250 copied to index 251; a pool for each index
    holding the outcome, bound to the bad snapshot's digest. Must exit 2, name both indices,
    and create no output. Two sessions on one date stay legal (every fixture day has both)."""
    c = Case("same_date_two_sessions")
    check(c.dataset[250]["date"] == c.dataset[251]["date"] and
          c.dataset[250]["session"] != c.dataset[251]["session"], "A-R26-FIXTURE", "fixture day lacks two sessions")
    c.fill("A0", lambda t, y: excl(y, 2))
    rc = c.run()
    check(rc == 0, "A-R26-SESSION", f"two sessions of one date refused: rc={rc} {c.stderr}")
    ds = make_dataset()
    ds[251] = dict(ds[250])
    c = Case("repeated_draw", dataset=ds)
    c.fill("A0", lambda t, y: incl(y, 3))
    dk = lambda r: (r["target"]["combined_index"], r["target"]["date"], r["target"]["session"])
    check(dk(c.records[50])[0] == 250 and dk(c.records[51])[0] == 251 and
          dk(c.records[50])[1:] == dk(c.records[51])[1:] and
          all(r["dataset_sha256"] == c.ds_sha for r in c.records), "A-R26-FIXTURE", "fixture not as Beta built it")
    rc = c.run()
    check(rc == 2 and "dataset rows 250 and 251: repeated (date, session)" in c.stderr, "A-R26-REFUSE",
          f"rc={rc} {c.stderr}")
    check(not c.out.exists(), "A-R26-NOOUT", f"{c.out} was created")


@gate
def test_missing():
    c = Case("missing")
    c.fill("A0", lambda t, y: excl(y, 2))
    del c.records[100]
    rc = c.run()
    check(rc == 2 and "missing input is an error" in c.stderr, "A-MISSING", f"rc={rc} {c.stderr}")
    c = Case("missing_arm")
    c.arms["arms"].append({"arm_id": "A1", "config_id": "cfgA1", "upstream_unseen_from_index": 200})
    c.fill("A0", lambda t, y: excl(y, 2))
    rc = c.run()
    check(rc == 2 and "arm A1" in c.stderr, "A-MISSING-ARM", f"rc={rc} {c.stderr}")


@gate
def test_malformed():
    c = Case("malformed")
    c.fill("A0", lambda t, y: excl(y, 2))
    c.raw_lines = "".join(json.dumps(r) + "\n" for r in c.records) + "{bad json\n"
    rc = c.run()
    check(rc == 2 and "malformed JSON" in c.stderr, "A-MALFORMED", f"rc={rc} {c.stderr}")
    c = Case("blank_line")
    c.fill("A0", lambda t, y: excl(y, 2))
    c.raw_lines = "\n".join(json.dumps(r) for r in c.records[:5]) + "\n\n" + \
        "".join(json.dumps(r) + "\n" for r in c.records[5:])
    rc = c.run()
    check(rc == 2 and "blank line" in c.stderr, "A-MALFORMED", f"rc={rc} {c.stderr}")
    c = Case("wrong_schema")
    c.fill("A0", lambda t, y: excl(y, 2))
    c.records[0]["schema"] = "other"
    rc = c.run()
    check(rc == 2 and "schema" in c.stderr, "A-SCHEMA", f"rc={rc} {c.stderr}")
    c = Case("no_pools_file")
    c.fill("A0", lambda t, y: excl(y, 2))
    c.run()
    shutil.rmtree(c.out)
    (c.dir / "pools.jsonl").unlink()
    err = io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
        rc = G.main(c.argv)
    check(rc == 2 and "pools file not found" in err.getvalue(), "A-NOFILE", f"rc={rc} {err.getvalue()}")


@gate
def test_leak():
    c = Case("leak")
    c.arms["arms"][0]["upstream_unseen_from_index"] = 250
    c.fill("A0", lambda t, y: excl(y, 2))
    rc = c.run()
    check(rc == 2 and "(R2)" in c.stderr, "A-LEAK", f"rc={rc} {c.stderr}")


@gate
def test_cutoff():
    c = Case("cutoff")
    c.fill("A0", lambda t, y: excl(y, 2))
    c.records[30]["data_cutoff_index"] = c.records[30]["target"]["combined_index"] + 1
    rc = c.run()
    check(rc == 2 and "S176 Phase C" in c.stderr, "A-CUTOFF", f"rc={rc} {c.stderr}")


@gate
def test_dataset_sha():
    c = Case("dataset_sha")
    c.fill("A0", lambda t, y: excl(y, 2))
    c.ds_sha_bound = c.ds_sha
    c.run()
    shutil.rmtree(c.out)
    argv = list(c.argv)
    argv[argv.index("--dataset-sha256") + 1] = "a" * 64
    err = io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
        rc = G.main(argv)
    check(rc == 2 and "dataset sha256 mismatch" in err.getvalue(), "A-DSHA", f"rc={rc} {err.getvalue()}")
    bad = make_dataset()
    bad[5]["draw"] = 1000
    c = Case("dataset_bad_draw", dataset=bad)
    c.fill("A0", lambda t, y: excl(y, 2))
    rc = c.run()
    check(rc == 2 and "dataset row 5" in c.stderr, "A-DS-RANGE", f"rc={rc} {c.stderr}")


@gate
def test_outdir():
    c = Case("outdir")
    c.fill("A0", lambda t, y: excl(y, 2))
    c.out_existing = c.dir / "out"
    c.out_existing.mkdir()
    rc = c.run()
    check(rc == 2 and "already exists" in c.stderr, "A-OUTDIR", f"rc={rc} {c.stderr}")
    argv = list(c.argv)
    target = "/tmp/e2e_grader_selftest_must_not_exist"
    check(not Path(target).exists(), "A-FIXTURE", "tmp probe path exists")
    argv[argv.index("--out-dir") + 1] = target
    err = io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
        rc = G.main(argv)
    check(rc == 2 and "is under /tmp" in err.getvalue() and not Path(target).exists(),
          "A-OUTDIR-TMP", f"rc={rc} {err.getvalue()}")


@gate
def test_pb_exact():
    ps = [0.02] * 50
    check(np.allclose(G.pb_pmf(ps), binom.pmf(np.arange(51), 50, 0.02), atol=1e-14),
          "A-PB-BINOM", "PB with equal p must equal binomial")
    rng = np.random.default_rng(3)
    ps = rng.uniform(0.001, 0.4, 10).tolist()
    brute = np.zeros(11)
    for bits in product((0, 1), repeat=10):
        pr = 1.0
        for b, p in zip(bits, ps):
            pr *= p if b else (1 - p)
        brute[sum(bits)] += pr
    check(np.allclose(G.pb_pmf(ps), brute, atol=1e-13), "A-PB-BRUTE", "PB vs enumeration")
    for h in range(0, 12):
        check(abs(G.pb_upper_tail(ps, h) - brute[h:].sum()) < 1e-12, "A-PTAIL", f"h={h}")
    ps = [0.02] * 300 + [0.1] * 100
    lo, up = G.lift_bounds(ps, 30, 0.025, 0.025)
    check(abs(G.pb_upper_tail(G._scaled(ps, lo), 30) - 0.025) < 1e-6, "A-LIFT", f"lo={lo}")
    check(abs(G.pb_lower_tail(G._scaled(ps, up), 30) - 0.025) < 1e-6, "A-LIFT", f"up={up}")
    check(lo < 30 / sum(ps) < up, "A-LIFT", f"{lo} {up}")
    lo0, _ = G.lift_bounds(ps, 0, 0.05, None)
    check(lo0 == 0.0, "A-LIFT", "H=0 lower bound must be 0")


@gate
def test_bonferroni_and_family():
    c = Case("bonferroni")
    c.arms["arms"] = [{"arm_id": a, "config_id": f"cfg{a}", "upstream_unseen_from_index": 200}
                      for a in ("A0", "A1", "A2")]
    for a in ("A0", "A1", "A2"):
        c.fill(a, lambda t, y: incl(y, 10) if t % 25 == 0 else excl(y, 10))
    rc = c.run()
    check(rc == 0, "A-RC0", f"rc={rc} {c.stderr}")
    r = c.result("A0")
    check(r["m"] == 3, "A-BONF", f"m={r['m']}")
    check(abs(r["p_adjusted"] - min(1.0, 3 * r["stats"]["p_one_sided"])) < 1e-15, "A-BONF",
          f"{r['p_adjusted']} vs 3*{r['stats']['p_one_sided']}")
    c2 = Case("family_all")
    c2.arms["arms"] = [{"arm_id": a, "config_id": f"cfg{a}", "upstream_unseen_from_index": 200}
                       for a in ("A0", "A1")]
    c2.arms["comparisons"] = [{"arm": "A1", "baseline": "A0"}]
    for a in ("A0", "A1"):
        c2.fill(a, lambda t, y: excl(y, 4))
    rc = c2.run()
    check(rc == 0 and c2.result("A0")["m"] == 3, "A-M-ALL", f"rc={rc} {c2.stderr}")
    c3 = Case("family_arms_only")
    c3.arms = c2.arms
    for a in ("A0", "A1"):
        c3.fill(a, lambda t, y: excl(y, 4))
    rc = c3.run("--multiplicity-family", "arms_only")
    check(rc == 0 and c3.result("A0")["m"] == 2, "A-M-ARMS", f"rc={rc} {c3.stderr}")


@gate
def test_holm():
    adj = G.adjust_pvalues([0.01, 0.04, 0.03], "holm")
    check(np.allclose(adj, [0.03, 0.06, 0.06]), "A-HOLM", str(adj))
    adj = G.adjust_pvalues([0.01, None, 0.03], "bonferroni")
    check(adj[1] is None and np.isclose(adj[0], 0.03) and np.isclose(adj[2], 0.09), "A-BONF-NONE", str(adj))


@gate
def test_uniform_control():
    """Uniform-random control lands inside the declared tolerance, and is deterministic
    across worker counts. k=2 on 3,000 draws makes a k-1 defect visible (z >> 4)."""
    ds = make_dataset(n=3200, seed=11)
    c = Case("uniform_control", dataset=ds)
    c.gb["frequency_control_block"] = {"start_index": 0, "end_index": 200}
    c.fill("A0", lambda t, y: excl(y, 2))
    rc = c.run(reps=100, workers=3)
    check(rc == 0, "A-RCTRL-PASS", f"rc={rc} {c.stderr}")
    u = c.summary["controls"]["A0"]["uniform_random"]
    check(u["passed"] and abs(u["z"]) <= u["tolerance_z"], "A-RCTRL-PASS", str({k: u[k] for k in ("mean_hits", "expected_hits", "z")}))
    check(u["empirical_rejection_rate"] <= u["exact_test_size"] + 4 * np.sqrt(max(u["exact_test_size"], 1e-9) / u["reps"]) + 1e-9,
          "A-RCTRL-SIZE", str(u["empirical_rejection_rate"]))
    rc = c.run(reps=100, workers=1, out_name="out_w1")
    check(rc == 0 and c.summary["controls"]["A0"]["uniform_random"]["rep_hits"] == u["rep_hits"],
          "A-RCTRL-DET", "control replicate hits must not depend on worker count")


@gate
def test_frequency_control():
    ds = make_dataset()
    for i in range(0, 200):
        ds[i]["draw"] = 7 if i % 3 else 8
    ds[0]["draw"] = 9
    c = Case("frequency", dataset=ds)
    c.fill("A0", lambda t, y: excl(y, 2))
    rc = c.run()
    check(rc == 0, "A-RC0", f"rc={rc} {c.stderr}")
    f = c.summary["controls"]["A0"]["frequency"]
    check(f["top_values"][:3] == [7, 8, 9] and f["top_values"][3] == 0, "A-FREQ-ORDER", str(f["top_values"][:5]))
    want = sum(1 for t in range(200, 400) if ds[t]["draw"] in (7, 8))
    check(f["hits"] == want and f["n_block_rows"] == 200, "A-FREQ-HITS", f"{f['hits']} vs {want}")
    c = Case("frequency_causal")
    c.gb["frequency_control_block"] = {"start_index": 0, "end_index": 201}
    c.fill("A0", lambda t, y: excl(y, 2))
    rc = c.run()
    check(rc == 2 and "not causal" in c.stderr, "A-FREQ-CAUSAL", f"rc={rc} {c.stderr}")


@gate
def test_abstention():
    c = Case("abst_ok")
    c.abst.update(allow_abstention=True, allowed_reasons=["no_model"])
    c.fill("A0", lambda t, y: excl(y, 2))
    c.records[10] = c.rec("A0", 210, [], abstain=True, abstain_reason="no_model")
    rc = c.run()
    check(rc == 0, "A-RC0", f"rc={rc} {c.stderr}")
    r = c.result("A0")
    check(r["n_abstained"] == 1 and r["stats"]["n_graded"] == 199 and r["n_eligible"] == 200,
          "A-ABST-DENOM", str((r["n_abstained"], r["stats"]["n_graded"], r["n_eligible"])))
    check("abstained 1 of 200 eligible" in r["headline"], "A-ABST-DENOM", r["headline"])
    c = Case("abst_reason")
    c.abst.update(allow_abstention=True, allowed_reasons=["no_model"])
    c.fill("A0", lambda t, y: excl(y, 2))
    c.records[10] = c.rec("A0", 210, [], abstain=True, abstain_reason="felt_like_it")
    rc = c.run()
    check(rc == 2 and "not in the declared list" in c.stderr, "A-ABST-REASON", f"rc={rc} {c.stderr}")
    c = Case("abst_deny")
    c.fill("A0", lambda t, y: excl(y, 2))
    c.records[10] = c.rec("A0", 210, [], abstain=True, abstain_reason="no_model")
    rc = c.run()
    check(rc == 2 and "abstention not allowed" in c.stderr, "A-ABST-DENY", f"rc={rc} {c.stderr}")
    c = Case("empty_pool")
    c.fill("A0", lambda t, y: excl(y, 2))
    c.records[10] = c.rec("A0", 210, [])
    rc = c.run()
    check(rc == 2 and "empty pool" in c.stderr, "A-EMPTY", f"rc={rc} {c.stderr}")


@gate
def test_null_first():
    c = Case("null_first")
    c.arms["arms"] = [{"arm_id": "A9", "config_id": "cfgA9", "upstream_unseen_from_index": 200},
                      {"arm_id": "Z0", "config_id": "cfgZ0", "upstream_unseen_from_index": 200}]
    c.fill("A9", lambda t, y: incl(y, 1))
    c.fill("Z0", lambda t, y: excl(y, 1))
    rc = c.run()
    check(rc == 0, "A-RC0", f"rc={rc} {c.stderr}")
    check([r["arm_id"] for r in c.summary["results"]] == ["Z0", "A9"], "A-NULL-FIRST",
          str([r["arm_id"] for r in c.summary["results"]]))
    check(c.result("A9")["above_chance"] and not c.result("Z0")["above_chance"], "A-NULL-FIRST", "")
    check(len(c.summary["null_results_first"]) == 1 and c.summary["null_results_first"][0].startswith("Z0: 0 hits of 200 graded draws"),
          "A-DENOM-HEADLINE", str(c.summary["null_results_first"]))
    md = (c.out / "SUMMARY.md").read_text()
    arm_sec = md.split("## Arm results (null results first)")[1]
    check(arm_sec.index("- Z0:") < arm_sec.index("- A9:"), "A-NULL-FIRST", "SUMMARY.md order")


@gate
def test_comparison():
    c = Case("comparison")
    c.arms["arms"] = [{"arm_id": a, "config_id": f"cfg{a}", "upstream_unseen_from_index": 200}
                      for a in ("A0", "A1")]
    c.arms["comparisons"] = [{"arm": "A1", "baseline": "A0"}]
    c.fill("A0", lambda t, y: incl(y, 5) if t % 50 == 0 else excl(y, 5))
    c.fill("A1", lambda t, y: incl(y, 5) if t % 10 == 0 else excl(y, 5))
    rc = c.run()
    check(rc == 0, "A-RC0", f"rc={rc} {c.stderr}")
    cmp_ = c.summary["comparisons"][0]
    b = sum(1 for t in range(200, 400) if t % 10 == 0 and t % 50 != 0)
    cc = 0
    check(cmp_["tested"] and cmp_["discordant_arm_only"] == b and cmp_["discordant_baseline_only"] == cc,
          "A-MCNEMAR", str(cmp_))
    check(abs(cmp_["p_one_sided"] - binom.sf(b - 1, b + cc, 0.5)) < 1e-15, "A-MCNEMAR", str(cmp_["p_one_sided"]))
    c = Case("comparison_unequal")
    c.arms["arms"] = [{"arm_id": a, "config_id": f"cfg{a}", "upstream_unseen_from_index": 200}
                      for a in ("A0", "A1")]
    c.arms["comparisons"] = [{"arm": "A1", "baseline": "A0"}]
    c.fill("A0", lambda t, y: excl(y, 5))
    c.fill("A1", lambda t, y: excl(y, 6))
    rc = c.run()
    cmp_ = c.summary["comparisons"][0] if c.summary else {}
    check(rc == 0 and cmp_.get("tested") is False and "unequal budget" in cmp_.get("reason", "")
          and cmp_.get("sum_k_arm") == 1200 and cmp_.get("sum_k_baseline") == 1000,
          "A-UNEQUAL", str(cmp_))


@gate
def test_previously_inspected():
    c = Case("prev_inspected")
    c.arms["arms"][0]["previously_inspected_blocks"] = [[300, 350]]
    c.fill("A0", lambda t, y: excl(y, 2))
    rc = c.run()
    check(rc == 2 and "(R13)" in c.stderr, "A-R13", f"rc={rc} {c.stderr}")


@gate
def test_max_k_and_sessions():
    c = Case("max_k")
    c.arms["arms"][0]["max_k"] = 20
    c.fill("A0", lambda t, y: excl(y, 2))
    c.records[0] = c.rec("A0", 200, list(range(21)))
    rc = c.run()
    check(rc == 2 and "outside arm A0's declared [1, 20]" in c.stderr, "A-MAXK", f"rc={rc} {c.stderr}")
    c = Case("sessions_midday")
    c.gb["sessions"] = ["midday"]
    c.fill("A0", lambda t, y: excl(y, 2))
    rc = c.run()
    check(rc == 2 and "outside the grade block" in c.stderr, "A-OOB", f"rc={rc} {c.stderr}")
    rc = c.run("--out-of-block-records", "exclude_and_count", out_name="out2")
    r = c.result("A0") if c.summary else {}
    check(rc == 0 and r["stats"]["n_graded"] == 100 and r["n_excluded_out_of_block"] == 100,
          "A-OOB-COUNT", f"rc={rc} {c.stderr}")


@gate
def test_cli_subprocess():
    c = Case("cli")
    c.fill("A0", lambda t, y: excl(y, 2))
    c.run()  # writes the input files
    shutil.rmtree(c.out)
    argv = [a for a in c.argv if a != "--no-provenance"]
    p = subprocess.run([sys.executable, "-I", str(GRADER_PATH)] + argv + ["--repo-root", str(REPO)],
                       capture_output=True, text=True)
    check(p.returncode == 0, "A-CLI", p.stderr)
    s = json.loads((c.out / "summary.json").read_text())
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    check(s["provenance"]["source_commit"] == head and s["provenance"]["authoritative"] is True,
          "A-PROV", str(s["provenance"].get("source_commit")))
    check(s["provenance"]["grader_sha256"] == G.sha256_file(GRADER_PATH), "A-PROV", "grader sha")
    p = subprocess.run([sys.executable, "-I", str(GRADER_PATH)] + argv + ["--repo-root", str(REPO)],
                       capture_output=True, text=True)
    check(p.returncode == 2 and "already exists" in p.stderr, "A-CLI-ERR", p.stderr)
    argv2 = list(argv)
    argv2[argv2.index("--out-dir") + 1] = str(c.dir / "out_noprov")
    p = subprocess.run([sys.executable, "-I", str(GRADER_PATH)] + argv2 + ["--repo-root", str(REPO / "no_such_dir_e2e_grader")],
                       capture_output=True, text=True)
    check(p.returncode == 2 and "provenance" in p.stderr, "A-PROV-FATAL", p.stderr)


# ------------------------------------------- TASK_LOOP_GRADE gates (R22, R23), 2026-11-23 --
REFERENCE = Path(os.environ.get("E2E_GRADER_REFERENCE", REPO / "tests" / "fixtures" /
                                "e2e_grader_a4d96fe7.py.txt"))   # tracked; was untracked data/ (2026-11-24)
REFERENCE_SHA = "a4d96fe7ac441a8befc93f55a638eb70e36e7d0cd70f24eaeb60d08b7a52169a"
NEW_KEYS_TOP = ("generated_utc", "provenance", "mode")
NEW_KEYS_CFG = ("mode", "per_draw_detail", "rank_summary", "rank_source", "rank_test_reps",
                "rank_test_seed", "per_session_report")


def _subrun(path, argv, out):
    a = list(argv)
    a[a.index("--out-dir") + 1] = str(out)
    p = subprocess.run([sys.executable, "-I", str(path)] + a, capture_output=True, text=True)
    return p


def _normal_summary(s):
    s = json.loads(json.dumps(s))
    for k in NEW_KEYS_TOP:
        s.pop(k, None)
    for k in NEW_KEYS_CFG:
        s["config"].pop(k, None)
        s["defaults"].pop(k, None)
    return s


def _equiv_case(name, setup, extra):
    c = Case(name)
    setup(c)
    c.run(*extra)                         # writes the input files
    shutil.rmtree(c.out)
    ref = _subrun(REFERENCE, c.argv, c.dir / "out_ref")
    new = _subrun(GRADER_PATH, c.argv, c.dir / "out_new")
    return c, ref, new


@gate
def test_counted_default_equivalence():
    """R22: with defaults (counted, no detail) the grader's outputs equal the authorised
    a4d96fe7 grader's, apart from the timestamp, provenance and the recorded mode fields, on
    datasets with unique (date, session) (A-EQUIV-DOMAIN); on a repeat the reference grades and
    the current grader refuses (R26, A-EQUIV-R26)."""
    check(REFERENCE.is_file() and G.sha256_file(REFERENCE) == REFERENCE_SHA, "A-EQUIV-REF",
          f"reference grader missing or changed: {REFERENCE}")

    def setup_a(c):
        c.gb["end_index"] = 410                                   # pending 400..409
        c.arms["arms"].append({"arm_id": "A1", "config_id": "cfgA1",
                               "upstream_unseen_from_index": 200})
        c.arms["comparisons"] = [{"arm": "A1", "baseline": "A0"}]
        c.abst["allow_abstention"] = True
        for t in range(200, 410):
            y = c.y(t) if t < 400 else 5
            c.records.append(c.rec("A0", t, incl(y, 3) if t % 9 == 0 else excl(y, 3)))
            if t % 13 == 0:
                c.records.append(c.rec("A1", t, [], abstain=True, abstain_reason="gate"))
            else:
                c.records.append(c.rec("A1", t, incl(y, 3) if t % 5 == 0 else excl(y, 3)))

    def setup_b(c):
        c.gb["sessions"] = ["midday"]
        c.fill("A0", lambda t, y: incl(y, 7) if t % 4 == 0 else excl(y, 7))
        c.records.append(c.rec("A0", 202, excl(c.y(202), 7)))     # duplicate target
    cases = [("equiv_a", setup_a, ()),
             ("equiv_b", setup_b, ("--out-of-block-records", "exclude_and_count",
                                   "--duplicate-rule", "first_in_file"))]
    for name, setup, extra in cases:
        c, ref, new = _equiv_case(name, setup, extra)
        check(ref.returncode == 0 and new.returncode == 0, "A-EQUIV-RC",
              f"{name}: ref {ref.returncode} {ref.stderr[-300:]} new {new.returncode} {new.stderr[-300:]}")
        ro, no = c.dir / "out_ref", c.dir / "out_new"
        check((ro / "per_draw.jsonl").read_bytes() == (no / "per_draw.jsonl").read_bytes(),
              "A-EQUIV-PERDRAW", f"{name}: per_draw.jsonl differs")
        rs = _normal_summary(json.loads((ro / "summary.json").read_text()))
        ns = _normal_summary(json.loads((no / "summary.json").read_text()))
        check(rs == ns, "A-EQUIV-SUMMARY",
              f"{name}: summary differs in keys {[k for k in set(rs) | set(ns) if rs.get(k) != ns.get(k)]}")
        rm = (ro / "SUMMARY.md").read_text().splitlines()
        nm = (no / "SUMMARY.md").read_text().splitlines()
        keep = lambda L: [x for i, x in enumerate(L) if i != 0 and not x.startswith("Instrument:")]
        check(keep(rm) == keep(nm), "A-EQUIV-MD", f"{name}: SUMMARY.md differs")
        check(sorted(os.listdir(ro)) == sorted(os.listdir(no)), "A-EQUIV-FILES",
              f"{name}: {sorted(os.listdir(ro))} vs {sorted(os.listdir(no))}")
        check(ref.stdout.replace(str(ro), "OUT") == new.stdout.replace(str(no), "OUT"),
              "A-EQUIV-STDOUT", f"{name}: stdout differs")
        check(json.loads((no / "summary.json").read_text())["mode"] == "COUNTED", "A-EQUIV-MODE",
              "counted summary must record mode COUNTED")
        keys = [(r["date"], r["session"]) for r in c.dataset]
        check(len(set(keys)) == len(keys), "A-EQUIV-DOMAIN", f"{name}: equivalence case has a repeated draw")
    # R26: the one intended difference. On a dataset with a repeated (date, session) the
    # reference graded both rows; the current grader refuses (exit 2, nothing written).
    ds = make_dataset()
    ds[251] = dict(ds[250])
    c2 = Case("equiv_r26_dup", dataset=ds)
    c2.fill("A0", lambda t, y: incl(y, 3))
    c2.run()
    shutil.rmtree(c2.out, ignore_errors=True)
    ref = _subrun(REFERENCE, c2.argv, c2.dir / "out_ref")
    new = _subrun(GRADER_PATH, c2.argv, c2.dir / "out_new")
    check(ref.returncode == 0 and new.returncode == 2 and not (c2.dir / "out_new").exists(),
          "A-EQUIV-R26", f"repeated draw: ref rc {ref.returncode}, new rc {new.returncode} {new.stderr[-200:]}")


def _inspected_case(name):
    c = Case(name)
    c.arms["arms"][0]["previously_inspected_blocks"] = [[250, 300]]
    c.fill("A0", lambda t, y: excl(y, 2))
    return c


@gate
def test_dev_mode_explicit_r13():
    c = _inspected_case("dev_r13")
    rc = c.run()
    check(rc == 2 and "(R13)" in c.stderr and c.summary is None, "A-DEV-DEFAULT-R13",
          f"default mode must stay counted and refuse an inspected block: rc={rc} {c.stderr}")
    rc = c.run("--mode", "counted", out_name="out_c")
    check(rc == 2 and "(R13)" in c.stderr, "A-DEV-DEFAULT-R13", f"explicit counted: rc={rc}")
    rc = c.run("--mode", "development", out_name="out_d")
    check(rc == 0, "A-DEV-RC", f"rc={rc} {c.stderr}")
    r = c.result("A0")
    check(r.get("previously_inspected_overlaps") == [[250, 300]], "A-DEV-R13-RECORDED", str(r.get("previously_inspected_overlaps")))
    check(c.summary["mode"] == "DEVELOPMENT", "A-DEV-STAMP-SUMMARY", c.summary.get("mode"))


@gate
def test_dev_mode_stamps():
    c = _inspected_case("dev_stamps")
    c.records.append(c.rec("A0", 400, [1, 2], ))                   # pending target
    c.gb["end_index"] = 401
    rc = c.run("--mode", "development", "--per-draw-detail", "on", "--run-label", "wf origin 2")
    check(rc == 0, "A-DEV-RC", f"rc={rc} {c.stderr}")
    check(all(r.get("mode") == "DEVELOPMENT" for r in c.rows) and len(c.rows) == 201,
          "A-DEV-STAMP-ROWS", "every per_draw row must carry mode DEVELOPMENT")
    det = [json.loads(l) for l in (c.out / "per_draw_detail.jsonl").read_text().splitlines()]
    check(all(r.get("mode") == "DEVELOPMENT" for r in det) and len(det) == 201,
          "A-DEV-STAMP-DETAIL", "every detail row must carry mode DEVELOPMENT")
    sums = (c.out / "SHA256SUMS").read_text().split()
    check((c.out / "DEVELOPMENT").is_file() and "DEVELOPMENT" in sums
          and sums[sums.index("DEVELOPMENT") - 1] == G.sha256_file(c.out / "DEVELOPMENT"),
          "A-DEV-MARKER", "DEVELOPMENT marker file missing or not in SHA256SUMS")
    md = (c.out / "SUMMARY.md").read_text()
    check(md.startswith("# DEVELOPMENT grade summary, NOT COUNTED"), "A-DEV-MD", md[:80])
    check(c.summary["run_label"] == "DEVELOPMENT: wf origin 2"
          and c.summary["status"].startswith("DEVELOPMENT"), "A-DEV-STAMP-SUMMARY", c.summary["run_label"])
    heads = [l for l in c.stdout.splitlines() if l.startswith("A0") or "A0:" in l]
    check(heads and all(h.startswith("DEVELOPMENT, NOT A COUNTED VERDICT.") for h in heads),
          "A-DEV-HEAD", str(heads))


@gate
def test_dev_never_counted_verdict():
    """A planted strong arm: counted mode says ABOVE CHANCE; development never does."""
    c = Case("dev_verdict")
    c.fill("A0", lambda t, y: incl(y, 3) if t % 3 == 0 else excl(y, 3))
    rc = c.run(out_name="out_counted")
    check(rc == 0 and c.result("A0")["above_chance"] is True, "A-DEV-SANITY",
          "fixture must be above chance in counted mode")
    rc = c.run("--mode", "development", out_name="out_dev")
    check(rc == 0, "A-DEV-RC", f"rc={rc} {c.stderr}")
    r = c.result("A0")
    check(r["above_chance"] is None and r["counted_verdict"] is None and c.summary["above_chance"] == []
          and r["development_above_background"] is True and len(c.summary["development_above_background"]) == 1,
          "A-DEV-NO-VERDICT-FIELD", str({k: r.get(k) for k in ("above_chance", "counted_verdict",
                                                                 "development_above_background")}))
    texts = [(c.out / "summary.json").read_text(), (c.out / "SUMMARY.md").read_text(), c.stdout]
    check(not any("ABOVE CHANCE" in x or "NULL: not above chance" in x for x in texts),
          "A-DEV-NO-VERDICT-TEXT", "counted verdict wording found in a development output")


@gate
def test_counted_refuses_dev_block():
    c = Case("dev_block")
    c.gb["mode"] = "development"
    c.fill("A0", lambda t, y: excl(y, 2))
    rc = c.run()
    check(rc == 2 and "cannot be graded in counted mode" in c.stderr and c.summary is None,
          "A-DEV-BLOCK", f"rc={rc} {c.stderr}")
    rc = c.run("--mode", "development", out_name="out_d")
    check(rc == 0 and c.summary["grade_block"]["mode"] == "development", "A-DEV-BLOCK-OK", f"rc={rc} {c.stderr}")
    c.gb["mode"] = "sometimes"
    rc = c.run("--mode", "development", out_name="out_bad")
    check(rc == 2 and "'mode' must be one of" in c.stderr, "A-DEV-BLOCK-VALUE", f"rc={rc} {c.stderr}")


@gate
def test_per_draw_detail():
    """R23 on a fixture with known positions, repeats, two arms, pending, abstained, excluded."""
    c = Case("detail")
    c.gb["sessions"] = ["midday"]
    c.gb["end_index"] = 402
    c.abst["allow_abstention"] = True
    c.arms["arms"].append({"arm_id": "A1", "config_id": "cfgA1", "upstream_unseen_from_index": 200})
    want = {}
    for t in range(200, 402):
        y = c.y(t) if t < 400 else 7
        if t % 3 == 0:
            j = t % 5                                 # drawn number at position j, again at j+2
            pool = excl(y, 6)
            pool.insert(j, y)
            pool.insert(j + 2, y)
            want[t] = (j + 1, 2)
        else:
            pool = excl(y, 6)
            want[t] = (None, 0)
        c.records.append(c.rec("A0", t, pool, pool_order="vote_rank"))
        if t == 204:
            c.records.append(c.rec("A1", t, [], abstain=True, abstain_reason="gate"))
        else:
            c.records.append(c.rec("A1", t, excl(y, 4)))
    rc = c.run("--out-of-block-records", "exclude_and_count", out_name="out_off")
    check(rc == 0 and not (c.out / "per_draw_detail.jsonl").exists()
          and "per_draw_detail.jsonl" not in (c.out / "SHA256SUMS").read_text(),
          "A-DETAIL-OFF", "default must write no detail file")
    rc = c.run("--out-of-block-records", "exclude_and_count", "--per-draw-detail", "on", out_name="out_on")
    check(rc == 0, "A-DETAIL-RC", f"rc={rc} {c.stderr}")
    det = [json.loads(l) for l in (c.out / "per_draw_detail.jsonl").read_text().splitlines()]
    by = {(r["arm_id"], r["combined_index"]): r for r in det}
    check(len(det) == 2 * 202 and {a for a, _ in by} == {"A0", "A1"}, "A-DETAIL-ARMS", str(len(det)))
    for t in range(200, 400):
        r = by[("A0", t)]
        if c.dataset[t]["session"] != "midday":
            check(r["status"] == "excluded_out_of_block" and r["drawn"] is None and r["vote_rank"] is None,
                  "A-DETAIL-EXCL", str(r))
            continue
        check(r["date"] == c.dataset[t]["date"] and r["session"] == "midday" and r["status"] == "graded",
              "A-DETAIL-ID", str(r))
        check(r["drawn"] == c.y(t), "A-DETAIL-DRAWN", str(r))
        check(r["k"] == (7 if want[t][0] else 6) and r["k_raw"] == (8 if want[t][0] else 6),
              "A-DETAIL-K", str(r))
        check(r["vote_rank"] == want[t][0], "A-DETAIL-RANK", f"t={t} {r['vote_rank']} != {want[t][0]}")
        check(r["vote_count"] == want[t][1], "A-DETAIL-COUNT", f"t={t} {r['vote_count']} != {want[t][1]}")
        check(r["hit"] == (1 if want[t][0] else 0), "A-DETAIL-HIT", str(r))
        check(r["pool_order"] == "vote_rank" and "mode" not in r, "A-DETAIL-ORDER", str(r))
    for t in (400,):
        r = by[("A0", t)]
        check(r["status"] == "pending" and r["drawn"] is None and r["hit"] is None, "A-DETAIL-PENDING", str(r))
    r = by[("A1", 204)]
    check(r["status"] == "abstained" and r["drawn"] is None and r["k"] == 0, "A-DETAIL-ABST", str(r))
    sums = (c.out / "SHA256SUMS").read_text().split()
    check(sums[sums.index("per_draw_detail.jsonl") - 1] == G.sha256_file(c.out / "per_draw_detail.jsonl")
          and c.summary["outputs"]["per_draw_detail.jsonl"]["lines"] == len(det), "A-DETAIL-SHA", "detail sha")


# --------------------------------- TASK_LOOP_ADDENDUM_1 gates (R24 rank, R25 sessions) ----
def _perm_with(y, pos, seed):
    """A full ranking (permutation of 0..999) with value y at 0-based position pos."""
    rest = [v for v in np.random.default_rng(seed).permutation(1000).tolist() if v != y]
    return rest[:pos] + [y] + rest[pos:]


def _detail(c):
    return {(r["arm_id"], r["combined_index"]): r for r in
            (json.loads(l) for l in (c.out / "per_draw_detail.jsonl").read_text().splitlines())}


@gate
def test_rank_detail():
    """Per-draw rank for each source, ties as mid-ranks, tie size and source stated."""
    c = Case("rank_detail")
    c.gb["sessions"] = ["midday"]
    want = {}
    for t in range(200, 400):
        y = c.y(t)
        kind = (t // 2) % 3
        if kind == 0:                                  # strict full ranking
            pos = (t * 7) % 1000
            c.records.append(c.rec("A0", t, excl(y, 3), value_ranking=_perm_with(y, pos, t),
                                   rank_basis="fixture"))
            want[t] = (pos + 1.0, 1, "record_value_ranking")
        elif kind == 1:                                # scores with a 4-way tie at the top
            s = [0.0] * 1000
            top = [y, (y + 1) % 1000, (y + 2) % 1000, (y + 3) % 1000]
            for v in top:
                s[v] = 5.0
            c.records.append(c.rec("A0", t, excl(y, 3), value_scores=s))
            want[t] = (2.5, 4, "record_value_scores")
        else:                                          # pool-derived; y at first position 2
            pool = excl(y, 4)
            pool.insert(2, y)
            pool.insert(4, y)
            c.records.append(c.rec("A0", t, pool))
            want[t] = (3.0, 1, "pool_first_position_rest_tied")
    rc = c.run("--out-of-block-records", "exclude_and_count", "--per-draw-detail", "on")
    check(rc == 0, "A-RANK-RC", f"rc={rc} {c.stderr}")
    det = _detail(c)
    for t in range(200, 400, 2):
        r = det[("A0", t)]
        check(r["rank_source"] == want[t][2], "A-RANK-SOURCE", f"t={t} {r['rank_source']}")
        if want[t][2] == "record_value_ranking":
            check(r["rank"] == want[t][0] and r["rank_basis"] == "fixture", "A-RANK-PERM", f"t={t} {r['rank']} != {want[t][0]}")
        elif want[t][2] == "record_value_scores":
            check(r["rank"] == want[t][0], "A-RANK-TIES", f"t={t} {r['rank']} != 2.5 (mid-rank of 1..4)")
        else:
            check(r["rank"] == want[t][0], "A-RANK-POOL", f"t={t} {r['rank']} != 3")
        check(r["rank_tie_size"] == want[t][1], "A-RANK-TIESIZE", f"t={t} {r['rank_tie_size']}")
    # a miss in a pool-derived ordering: the drawn value is tied with all 995 unvoted values
    c2 = Case("rank_pool_miss")
    c2.gb["sessions"] = ["midday"]
    c2.fill("A0", lambda t, y: excl(y, 5))
    rc = c2.run("--out-of-block-records", "exclude_and_count", "--per-draw-detail", "on")
    r = _detail(c2)[("A0", 200)]
    check(rc == 0 and r["rank"] == (6 + 1000) / 2 and r["rank_tie_size"] == 995, "A-RANK-POOL",
          f"unvoted values must share the mid-rank 503: {r['rank']} tie {r['rank_tie_size']}")


@gate
def test_rank_summary():
    """Exact null moments, Monte Carlo direction, planted signal, all-tied orderings."""
    c = Case("rank_null")
    c.gb["sessions"] = ["midday"]
    for t in range(200, 400):
        c.records.append(c.rec("A0", t, excl(c.y(t), 2),
                               value_ranking=np.random.default_rng(1000 + t).permutation(1000).tolist()))
    rc = c.run("--out-of-block-records", "exclude_and_count", "--rank-summary", "on",
               "--rank-test-reps", "4000")
    check(rc == 0, "A-RANK-RC", f"rc={rc} {c.stderr}")
    rs = c.result("A0")["rank_summary"]
    check(rs["n"] == 100 and abs(rs["exact_null_mean_sum"] - 100 * 500.5) < 1e-6
          and abs(rs["expected_mean_rank"] - 500.5) < 1e-9, "A-RANK-MEAN", str(rs["exact_null_mean_sum"]))
    check(abs(rs["exact_null_sd_sum"] - math_sqrt(100 * (1000 ** 2 - 1) / 12)) < 1e-6, "A-RANK-SD",
          f"{rs['exact_null_sd_sum']} vs sqrt(n (N^2-1)/12)")
    check(abs(rs["p_lower_monte_carlo"] - rs["p_lower_normal_approx"]) < 0.03, "A-RANK-P",
          f"MC {rs['p_lower_monte_carlo']} vs normal {rs['p_lower_normal_approx']}")
    check(rs["control"]["passed"] and "rank_test" in c.summary["controls"]["A0"], "A-RANK-CTRL", str(rs["control"]))
    check("not in the multiplicity family" in rs["role"], "A-RANK-ROLE", rs["role"])
    # planted: drawn value at ranks 1..10 -> small p; at the bottom -> p_lower near 1
    for name, posfn, tag in (("rank_plant", lambda t: t % 10, "A-RANK-PLANT"),
                             ("rank_bottom", lambda t: 990 + t % 10, "A-RANK-DIR")):
        c = Case(name)
        c.gb["sessions"] = ["midday"]
        for t in range(200, 400):
            c.records.append(c.rec("A0", t, excl(c.y(t), 2), value_ranking=_perm_with(c.y(t), posfn(t), t)))
        rc = c.run("--out-of-block-records", "exclude_and_count", "--rank-summary", "on",
                   "--rank-test-reps", "2000")
        rs = c.result("A0")["rank_summary"] if c.summary else {}
        ok = rs and (rs["p_lower_monte_carlo"] <= 1 / 2001 + 1e-12 if tag == "A-RANK-PLANT"
                     else rs["p_lower_monte_carlo"] > 0.99)
        check(rc == 0 and ok, tag, f"rc={rc} p={rs.get('p_lower_monte_carlo')} mean={rs.get('mean_rank')}")
    # every value tied: each simulated sum equals the observed one, so p must be 1
    c = Case("rank_alltied")
    c.gb["sessions"] = ["midday"]
    for t in range(200, 400):
        c.records.append(c.rec("A0", t, excl(c.y(t), 2), value_scores=[1.0] * 1000))
    rc = c.run("--out-of-block-records", "exclude_and_count", "--rank-summary", "on", "--rank-test-reps", "500")
    rs = c.result("A0")["rank_summary"]
    check(rc == 0 and rs["p_lower_monte_carlo"] == 1.0 and rs["exact_null_sd_sum"] == 0
          and rs["ties"]["draws_where_drawn_value_is_tied"] == 100, "A-RANK-P-TIES", str(rs))


def math_sqrt(x):
    return float(np.sqrt(x))


@gate
def test_rank_not_in_verdict():
    """A strong rank signal with no hits: the counted verdict, m and the headline are unchanged."""
    c = Case("rank_verdict")
    c.gb["sessions"] = ["midday"]
    for t in range(200, 400):
        y = c.y(t)
        c.records.append(c.rec("A0", t, excl(y, 5), value_ranking=_perm_with(y, 5, t)))
    rc = c.run("--out-of-block-records", "exclude_and_count", out_name="off")
    base = c.result("A0")
    rc2 = c.run("--out-of-block-records", "exclude_and_count", "--rank-summary", "on",
                "--rank-test-reps", "1000", out_name="on")
    r = c.result("A0")
    check(rc == 0 and rc2 == 0 and r["rank_summary"]["p_lower_monte_carlo"] < 0.01
          and r["above_chance"] is False and r["m"] == 1 and r["p_adjusted"] == base["p_adjusted"]
          and r["headline"] == base["headline"] and c.summary["declared"]["m"] == 1,
          "A-RANK-VERDICT", f"rank must not enter the verdict: {r['above_chance']} m={r['m']}")


@gate
def test_rank_inputs():
    c = Case("rank_inputs")
    c.gb["sessions"] = ["midday"]
    c.fill("A0", lambda t, y: excl(y, 2))
    c.records[0] = c.rec("A0", 200, excl(c.y(200), 2), value_ranking=[0] * 1000)   # malformed
    rc = c.run("--out-of-block-records", "exclude_and_count", out_name="off")
    check(rc == 0, "A-RANK-OFF-COMPAT", f"with rank off a rank field is ignored, as today: rc={rc} {c.stderr}")
    rc = c.run("--out-of-block-records", "exclude_and_count", "--rank-summary", "on", out_name="on")
    check(rc == 2 and "value_ranking must list all" in c.stderr, "A-RANK-MALFORMED", f"rc={rc} {c.stderr}")
    c = Case("rank_reconly")
    c.gb["sessions"] = ["midday"]
    c.fill("A0", lambda t, y: excl(y, 2))
    rc = c.run("--out-of-block-records", "exclude_and_count", "--rank-summary", "on",
               "--rank-source", "record_only")
    check(rc == 2 and "record_only" in c.stderr, "A-RANK-RECONLY", f"rc={rc} {c.stderr}")


@gate
def test_per_session_report():
    c = Case("per_session")                       # grade block has both sessions
    c.fill("A0", lambda t, y: incl(y, 4) if t % 4 == 0 else (incl(y, 4) if t % 6 == 1 else excl(y, 4)))
    rc = c.run(out_name="off")
    check(rc == 0 and "per_session" not in c.result("A0"), "A-SESS-OFF", "default must not add per_session")
    base = c.result("A0")
    rc = c.run("--per-session-report", "on", out_name="on")
    r = c.result("A0")
    ps = r["per_session"]
    want_mid = sum(1 for t in range(200, 400) if t % 2 == 0 and t % 4 == 0)
    want_eve = sum(1 for t in range(200, 400) if t % 2 == 1 and t % 6 == 1)
    check(rc == 0 and ps["midday"]["hits"] == want_mid and ps["evening"]["hits"] == want_eve
          and ps["midday"]["n_graded"] == 100 and ps["evening"]["n_graded"] == 100, "A-SESS-SPLIT",
          f"{ps['midday'].get('hits')}/{want_mid} {ps['evening'].get('hits')}/{want_eve}")
    check(r["stats"]["hits"] == want_mid + want_eve and r["stats"] == base["stats"], "A-SESS-POOLED",
          "the arm's own result is the pooled one, unchanged")
    check(r["m"] == base["m"] and r["p_adjusted"] == base["p_adjusted"] and r["headline"] == base["headline"],
          "A-SESS-VERDICT", "per-session report must not change the verdict")


# ---------------------------------------------------------------------------- runner -----
def main():
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]
    WORK.mkdir(parents=True, exist_ok=True)
    sel = [t for t in TESTS if only is None or t.__name__ == only]
    if not sel:
        print(f"no test named {only}")
        return 2
    fails = 0
    t0 = time.time()
    for t in sel:
        try:
            t()
            print(f"PASS {t.__name__}", flush=True)
        except AssertionError as e:
            fails += 1
            print(f"FAIL {t.__name__}: AssertionError: {e}", flush=True)
        except Exception as e:
            fails += 1
            print(f"FAIL {t.__name__}: CRASH {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
    print(f"{len(sel) - fails}/{len(sel)} passed in {time.time() - t0:.1f}s; grader={GRADER_PATH}; "
          f"work={WORK}", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
