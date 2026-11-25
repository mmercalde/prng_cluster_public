#!/usr/bin/env python3
"""
Self-test for scripts/diagnostics/compare_settings.py (LL-4 first form). CPU only, synthetic
fixtures under <repo>/data/e2e_grader_selftest/ (never /tmp). Seeds are vehicles; nothing
here judges a seed.

Run:  python3 tests/test_compare_settings.py [--only NAME]
Env:  E2E_COMPARE_PATH (harness under test; default the real file), E2E_GRADER_PATH (grader;
      default the real file), E2E_SELFTEST_DIR (work dir).
Every assertion carries a tag "[A-CS-...]" for the mutation runner.
"""
import contextlib
import copy
import importlib.util
import inspect
import io
import json
import os
import shutil
import sys
import time
import traceback
from datetime import date, timedelta
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CS_PATH = Path(os.environ.get("E2E_COMPARE_PATH", REPO / "scripts" / "diagnostics" / "compare_settings.py"))
GRADER_PATH = Path(os.environ.get("E2E_GRADER_PATH", REPO / "scripts" / "diagnostics" / "e2e_grader.py"))
WORK = Path(os.environ.get("E2E_SELFTEST_DIR", REPO / "data" / "e2e_grader_selftest" /
                           time.strftime("cs_run_%Y%m%dT%H%M%S")))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


CS = _load(CS_PATH, "compare_settings_under_test")
G = _load(GRADER_PATH, "e2e_grader_for_cs_test")
TESTS = []


def gate(fn):
    TESTS.append(fn)
    return fn


def check(cond, tag, msg=""):
    if not cond:
        raise AssertionError(f"[{tag}] {msg}")


def make_dataset(n=400, seed=11):
    rng = np.random.default_rng(seed)
    d0 = date(2000, 1, 1)
    return [{"date": (d0 + timedelta(days=i // 2)).isoformat(),
             "session": "midday" if i % 2 == 0 else "evening", "draw": int(v)}
            for i, v in enumerate(rng.integers(0, 1000, n))]


def excl(y, k):
    return [(y + 1 + j) % 1000 for j in range(k)]


def incl(y, k):
    return [y] + [(y + 1 + j) % 1000 for j in range(k - 1)]


class Fixture:
    """Two variants A and B that differ only in step5.exclude_features (a setting), plus
    timing (bookkeeping) and identity fields."""

    def __init__(self, name):
        self.dir = WORK / name
        if self.dir.exists():
            shutil.rmtree(self.dir)
        self.dir.mkdir(parents=True)
        self.dataset = make_dataset()
        self.ds = self.dir / "dataset.json"
        self.ds.write_text(json.dumps(self.dataset))
        self.ds_sha = G.sha256_file(self.ds)
        self.gb = {"schema": G.SCHEMA_GRADE_BLOCK, "name": name, "start_index": 200, "end_index": 400,
                   "sessions": ["midday"],
                   "frequency_control_block": {"start_index": 0, "end_index": 200}}
        self.abst = {"schema": G.SCHEMA_ABSTENTION, "allow_abstention": False,
                     "allowed_reasons": None, "empty_pool": "error"}
        self.pool_fn = {"A": lambda t, y: incl(y, 5) if t % 6 == 0 else excl(y, 5),
                        "B": lambda t, y: incl(y, 5) if t % 4 == 0 else excl(y, 5)}
        self.extra_records = {"A": [], "B": []}
        self.run_records = {}
        for v in ("A", "B"):
            self.run_records[v] = {
                "arm_id": v, "config_id": f"cfg{v}",
                "dataset": {"sha256": self.ds_sha},
                "step3": {"seed": 1, "trials": 4},
                "step5": {"exclude_features": ["score"] if v == "A" else ["score", "global_x"],
                          "exclude_features_note": "same"},
                "vote": {"k": 5, "rule": "step6"},
                "timing": {"seconds": 10.0 if v == "A" else 12.5},
                "pools_jsonl": {"sha256": None}}
        self.decl = {"schema": CS.SCHEMA_DECL, "name": name,
                     "setting_fields": ["step5.exclude_features"],
                     "bookkeeping_fields": ["timing"],
                     "identity_fields": {"arm_id": "arm_id", "config_id": "config_id",
                                         "pools_sha256": "pools_jsonl.sha256"},
                     "variants": [], "comparisons": [{"arm": "B", "baseline": "A"}],
                     "grade_block": "grade_block.json", "abstention": "abstention.json",
                     "grader_options": {"control_reps": 20, "workers": 1,
                                        "out_of_block_records": "exclude_and_count"}}

    def write(self):
        for v in ("A", "B"):
            lines = []
            for t in range(200, 400):
                tgt = {"combined_index": t, "date": self.dataset[t]["date"],
                       "session": self.dataset[t]["session"]}
                lines.append({"schema": G.SCHEMA_POOL, "arm_id": v, "config_id": f"cfg{v}",
                              "dataset_sha256": self.ds_sha, "data_cutoff_index": t, "target": tgt,
                              "pool": self.pool_fn[v](t, self.dataset[t]["draw"])})
            lines += self.extra_records[v]
            p = self.dir / f"pools_{v}.jsonl"
            p.write_text("".join(json.dumps(r) + "\n" for r in lines))
            if self.run_records[v]["pools_jsonl"]["sha256"] is None:
                self.run_records[v]["pools_jsonl"]["sha256"] = G.sha256_file(p)
            rr = self.dir / f"run_record_{v}.json"
            rr.write_text(json.dumps(self.run_records[v], indent=1))
        for nm, obj in (("grade_block.json", self.gb), ("abstention.json", self.abst)):
            (self.dir / nm).write_text(json.dumps(obj))
        d = copy.deepcopy(self.decl)
        if not d["variants"]:
            d["variants"] = [{"arm_id": v, "config_id": f"cfg{v}",
                              "run_record": f"run_record_{v}.json",
                              "run_record_sha256": G.sha256_file(self.dir / f"run_record_{v}.json"),
                              "pools": f"pools_{v}.jsonl",
                              "pools_sha256": G.sha256_file(self.dir / f"pools_{v}.jsonl"),
                              "upstream_unseen_from_index": 200, "max_k": 20} for v in ("A", "B")]
        d["grade_block_sha256"] = G.sha256_file(self.dir / "grade_block.json")
        d["abstention_sha256"] = G.sha256_file(self.dir / "abstention.json")
        self.decl_path = self.dir / "declaration.json"
        self.decl_path.write_text(json.dumps(d, indent=1))
        self.decl_sha = G.sha256_file(self.decl_path)
        return d

    def run(self, *extra, out_name="out", decl_sha=None, grader_sha=None):
        self.out = self.dir / out_name
        argv = ["--declaration", str(self.decl_path), "--declaration-sha256", decl_sha or self.decl_sha,
                "--grader", str(GRADER_PATH), "--grader-sha256", grader_sha or G.sha256_file(GRADER_PATH),
                "--dataset", str(self.ds), "--dataset-sha256", self.ds_sha, "--out-dir", str(self.out),
                "--no-provenance"] + list(extra)
        err, outb = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(outb):
                rc = CS.main(argv)
        except Exception as e:
            rc = f"EXC:{type(e).__name__}:{e}"
        self.stderr, self.stdout = err.getvalue(), outb.getvalue()
        rp = self.out / "compare_record.json"
        self.record = json.loads(rp.read_text()) if rp.is_file() else None
        sp = self.out / "grade" / "summary.json"
        self.summary = json.loads(sp.read_text()) if sp.is_file() else None
        return rc


# ---------------------------------------------------------------------------- gates ------
@gate
def test_cs_signatures():
    """Harness discipline: the grader functions this harness calls have the expected shape."""
    want = {"read_pool_records": ["path"], "validate_pool_values": ["raw_pool", "draw_space", "where"],
            "parse_grade_block": ["obj"], "_check_out_dir": ["out_dir"]}
    for fn, params in want.items():
        got = list(inspect.signature(getattr(G, fn)).parameters)
        check(got == params, "A-CS-SIG", f"{fn}{got} != {params}")


@gate
def test_cs_happy():
    f = Fixture("cs_happy")
    f.write()
    rc = f.run()
    check(rc == 0, "A-CS-RC0", f"rc={rc} {f.stderr[-500:]}")
    r = f.record
    check(r["grader_calls"] == 1 and sorted(os.listdir(f.out)) == ["compare_record.json", "grade", "inputs"],
          "A-CS-ONCE", str(sorted(os.listdir(f.out))))
    comp = f.summary["comparisons"][0]
    check(comp["tested"] is True and comp["p_one_sided"] is not None and comp["equal_budget"] is True,
          "A-CS-MCNEMAR", str(comp))
    check(r["grade"]["comparisons"] == f.summary["comparisons"]
          and r["grade"]["null_results_first"] == f.summary["null_results_first"], "A-CS-PASSTHRU",
          "record must quote the grader's own results")
    want_hits = {"A": sum(1 for t in range(200, 400, 2) if t % 6 == 0),
                 "B": sum(1 for t in range(200, 400, 2) if t % 4 == 0)}
    got = {x["arm_id"]: x["stats"]["hits"] for x in f.summary["results"]}
    check(got == want_hits, "A-CS-HITS", f"{got} != {want_hits}")
    d = r["run_record_diff"]["B"]
    check([x["path"] for x in d["setting_differences"]] == ["step5.exclude_features[1]"]
          and "timing.seconds" in d["other_declared_differences"], "A-CS-DIFF-REPORT", str(d))
    check(r["inputs"]["pools_concat.jsonl"] == G.sha256_file(f.out / "inputs" / "pools_concat.jsonl")
          and r["declaration"]["sha256"] == f.decl_sha and r["mode"] == "COUNTED", "A-CS-BIND-REC", "")


@gate
def test_cs_hidden_diff():
    f = Fixture("cs_hidden")
    f.run_records["B"]["step3"]["seed"] = 2                      # undeclared difference
    f.write()
    rc = f.run()
    check(rc == 2 and "undeclared field" in f.stderr and "step3.seed" in f.stderr,
          "A-CS-HIDDEN-DIFF", f"rc={rc} {f.stderr}")
    check(not f.out.exists(), "A-CS-NOTHING-WRITTEN", "a refusal must write nothing")
    f = Fixture("cs_hidden_sep")
    f.run_records["B"]["step5"]["exclude_features_note"] = "changed"   # shares the prefix, not the path
    f.write()
    rc = f.run()
    check(rc == 2 and "step5.exclude_features_note" in f.stderr, "A-CS-COVER-SEP",
          f"a declared path must not cover a sibling sharing its prefix: rc={rc} {f.stderr}")


@gate
def test_cs_no_setting_diff():
    f = Fixture("cs_nodiff")
    f.run_records["B"]["step5"]["exclude_features"] = ["score"]
    f.write()
    rc = f.run()
    check(rc == 2 and "differ in no declared setting" in f.stderr, "A-CS-NO-DIFF", f"rc={rc} {f.stderr}")


@gate
def test_cs_overlap_groups():
    f = Fixture("cs_overlap")
    f.decl["bookkeeping_fields"] = ["timing", "step5"]
    f.write()
    rc = f.run()
    check(rc == 2 and "overlaps" in f.stderr, "A-CS-OVERLAP", f"rc={rc} {f.stderr}")


@gate
def test_cs_equal_k():
    f = Fixture("cs_equalk")
    f.pool_fn["B"] = lambda t, y: excl(y, 6) if t == 250 else (incl(y, 5) if t % 4 == 0 else excl(y, 5))
    f.write()
    rc = f.run()
    check(rc == 2 and "require_equal_k" in f.stderr and "target 250" in f.stderr, "A-CS-EQUAL-K",
          f"rc={rc} {f.stderr}")
    f.decl["require_equal_k"] = False
    f.write()
    rc = f.run(out_name="out_unequal")
    comp = f.summary["comparisons"][0] if f.summary else {}
    check(rc == 0 and comp.get("tested") is False and "unequal budget" in comp.get("reason", ""),
          "A-CS-UNEQUAL-PASS", f"rc={rc} {comp}")


@gate
def test_cs_third_arm():
    f = Fixture("cs_third")
    r = {"schema": G.SCHEMA_POOL, "arm_id": "C", "config_id": "cfgC", "dataset_sha256": "x",
         "data_cutoff_index": 200, "target": {"combined_index": 200, "date": "x", "session": "midday"},
         "pool": [1]}
    f.extra_records["B"].append(r)
    f.write()
    rc = f.run()
    check(rc == 2 and "COMPARE_SETTINGS REFUSED" in f.stderr and "undeclared arm" in f.stderr,
          "A-CS-THIRD-ARM", f"rc={rc} {f.stderr}")
    f = Fixture("cs_third_decl")
    d = f.write()
    d["variants"].append(dict(d["variants"][1], arm_id="C"))       # added after declaring
    f.decl_path.write_text(json.dumps(d, indent=1))
    rc = f.run()
    check(rc == 2 and "declaration sha256 mismatch" in f.stderr, "A-CS-DECL-SHA", f"rc={rc} {f.stderr}")


@gate
def test_cs_r13_and_development():
    f = Fixture("cs_r13")
    f.decl["previously_inspected_blocks"] = [[300, 320]]
    f.write()
    rc = f.run()
    check(rc == 2 and "(R13)" in f.stderr and f.record["grader_exit"] == 2 and f.summary is None,
          "A-CS-R13", f"rc={rc} {f.stderr[-400:]}")
    rc = f.run("--mode", "development", out_name="out_dev")
    check(rc == 0 and f.summary["mode"] == "DEVELOPMENT" and f.record["mode"] == "DEVELOPMENT"
          and f.summary["above_chance"] == [] and (f.out / "grade" / "DEVELOPMENT").is_file(),
          "A-CS-DEV", f"rc={rc} {f.stderr[-400:]}")


@gate
def test_cs_bindings():
    f = Fixture("cs_bind")
    f.write()
    rc = f.run(grader_sha="0" * 64)
    check(rc == 2 and "grader sha256 mismatch" in f.stderr, "A-CS-GRADER-SHA", f"rc={rc} {f.stderr}")
    (f.dir / "run_record_B.json").write_text(json.dumps(dict(f.run_records["B"], extra=1)))
    rc = f.run(out_name="out2")
    check(rc == 2 and "run record sha256 mismatch" in f.stderr, "A-CS-INPUT-SHA", f"rc={rc} {f.stderr}")
    f = Fixture("cs_ident")
    f.run_records["B"]["config_id"] = "cfgOTHER"
    f.write()
    rc = f.run()
    check(rc == 2 and "(C5)" in f.stderr, "A-CS-IDENTITY", f"rc={rc} {f.stderr}")


# ---------------------------------------------------------------------------- runner -----
def main():
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
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
    print(f"{len(sel) - fails}/{len(sel)} passed in {time.time() - t0:.1f}s; harness={CS_PATH}; "
          f"grader={GRADER_PATH}; work={WORK}", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
