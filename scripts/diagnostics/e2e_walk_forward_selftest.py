#!/usr/bin/env python3
"""
scripts/diagnostics/e2e_walk_forward_selftest.py -- self-test for e2e_walk_forward.py and the
shim's --target-field. CPU only, SYNTHETIC FIXTURE ONLY (no run H row is read). Everything is
written under <worktree>/data/e2e_wf_selftest_<stamp>/.

    ~/venvs/torch/bin/python3 -B scripts/diagnostics/e2e_walk_forward_selftest.py

Pins: the vote harness and the grader run from sha-verified SNAPSHOTS made at test start (their
agents edit them in place); the driver refuses if a pinned file changes during a run.

Gates:
  W1  full run on the fixture (360 rows, sieve end 20, L = 40, ~40-draw blocks), origins 1-3,
      lightgbm + xgboost, pools 20 and 60, sessions midday / evening / pooled, model choice
      prev_block_rank: rc 0; 54 table rows; DEVELOPMENT on every driver output and per-draw row;
      grader DEVELOPMENT marker everywhere; L1-L6 ok; train [0, b_i - L) and labels
      [b_i - L, b_i) (the train end moves); range starts at the earliest workable origin; cuts
      on date changes; origin 1 chooses by cv, origins 2-3 by the previous graded block; winner
      and every family frozen read-only before the first vote; counts per session view, pooled
      = midday + evening; rank re-derived independently
  W2  LEAK (perturbation): every draw at or after b_2 replaced; origins 1 and 2 identical in
      every field upstream of the grade (17 for origin 2), outcomes differ
  W2m three leaking mutants with runtime checks off; the W2 comparison sees each one
  W3  runtime checks, one mutant each: L1, L3, L4 (choice from the current block; vote with no
      frozen winner), L5, L6, counted mode refused by the grader (R22), table completeness,
      driver/grader rank mismatch
  W4  refusals (exit 2, nothing created)
  W5  resume: every step skipped after verification, same table
  W6  Step 5 target setting through the shim
  W8  ranker arms (ADDENDUM_2): exact groups (each survivor's vote = the vote harness's own pool, row
      for row, at every label draw), planted good survivors ranked on top for all four objectives,
      bounded sizes, label sha refusal, driver integration (L7, frozen, voted, graded), mutants
      relevance_shift and ranker_label_shift
  W9  grader R24 rank on every grade (equal to the driver's with ordering off); full-vote
      value ordering plumbing; the selfplay_window interface
  W10 pipeline switches (owner, TASK_LAND_WF_VOTE_20261125_ADDENDUM_1): for LL-1, LL-2 and LL-3,
      (a) the capability check refuses each switch on pipeline files that lack it (git show of
      --caps-without-rev, default HEAD), naming the switch and the file; (b) non-vacuity: the
      same requests pass on files that have it (--caps-with-root, default the e2e-map worktree);
      (c) the driver on THIS tree: refuses (exit 2, nothing created, switch and file named) when
      this tree lacks the switch, and records it as applied (--plan-only) when it has it;
      (d) the Step 3 harness run directly with --per-seed-sieve-features on refuses before its
      out-dir exists when this tree lacks LL-2
  W11 run H wrappers (lead ADDENDUM_4 Fix 1, Beta X-E2E-WF-VOTE R1 Q1): the REAL runH_commands
      scripts, in a tree built from this one whose Step 3 pipeline files lack LL-2 (git show of
      --caps-without-rev): 01_sieve_on exits non-zero having created no log, no .time file, no
      out-dir and no launch; non-vacuity: the same script in a tree whose files have LL-2
      (--caps-with-root) passes the preflight and reaches the launch; 00, 02, 03 launch on the
      files without LL-2. `taskset` is a stub that records its call, so nothing heavy runs.
      Needs the run H inputs under ~/skipsw_run_thr3030 (read only).
  W12 passthroughs (ADDENDUM_4 Fixes 2-3, Beta Q1/Q2): abbreviations refused by the driver
      (--per-seed-sieve-f on) and by the Step 3 harness and shim parsers; driver-set options
      refused in --opt VALUE and --opt=VALUE form (--survivors=PATH, --k-folds 2, --k-folds=2,
      --trials=3 via --nn-args, --out-dir via Step 3); --check-only writes nothing; Beta's
      regression: through the real shim and main's own optimizer parser body, a holdout_hits run
      reads the derived file (target 2, not the source's 0.045) with the declared folds, and a
      trailing --survivors=SRC is refused by the shim; one mutant per guard
  W7  no-write: git status of the main tree and the vote and grader worktrees unchanged, no
      new __pycache__ there, pinned files unchanged
"""
from __future__ import annotations

import sys as _sys
_sys.dont_write_bytecode = True

import argparse
import datetime as _dt
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WT = HERE.parents[1]
DRIVER = HERE / "e2e_walk_forward.py"
SHIM = HERE / "e2e_step5_shim.py"
MODULE5 = WT / "meta_prediction_optimizer_anti_overfit.py"
PY = sys.executable
VOTE = HERE / "e2e_vote_harness.py"     # this repo's copy (was the e2e-vote worktree)
GRADER = HERE / "e2e_grader.py"         # this repo's copy (was the e2e-grader worktree)
VCFG = WT / "prediction_generator_config.json"
MAIN = Path("~/distributed_prng_analysis").expanduser()
RESULTS = []


def gate(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ST3 = load(HERE / "e2e_step3_selftest.py", "e2e_step3_selftest")
WF = load(DRIVER, "e2e_walk_forward")


def git_status(root):
    return subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True,
                          text=True).stdout


def pycaches(root):
    return sorted(str(p) for p in Path(root).rglob("__pycache__")
                  if "data" not in p.relative_to(root).parts)


class Ctx:
    pass


def snapshot_vote(dst: Path):
    """Frozen copy of the vote harness that keeps its repo-relative imports: a real copy of
    scripts/diagnostics/e2e_vote_harness.py (REPO_ROOT = parents[2] of the resolved file, so
    dst) plus read-only symlinks to the vote worktree's other top-level entries. Its agent
    may keep editing the harness in place; the test runs the snapshot, pinned."""
    src_root = VOTE.parents[2]
    (dst / "scripts" / "diagnostics").mkdir(parents=True)
    shutil.copyfile(VOTE, dst / "scripts" / "diagnostics" / VOTE.name)
    caps = VOTE.parent / "e2e_pipeline_caps.py"     # the harness's LL-1 check (ADDENDUM_2) sits beside it
    if caps.is_file():
        shutil.copyfile(caps, dst / "scripts" / "diagnostics" / caps.name)
    for e in src_root.iterdir():
        if e.name in ("scripts", "data", "docs", "logs", "tests", ".git", "__pycache__"):
            continue
        os.symlink(e, dst / e.name)
    return dst / "scripts" / "diagnostics" / VOTE.name


FX_FRAME = {"--sieve-window-end-index": "20", "--min-train-draws": "100", "--label-draws": "40",
            "--block-length": "40"}     # fixture: 360 rows -> range [140, 360), 6 blocks


def driver_args(C, fx, out, origins="1,2,3", families="lightgbm,xgboost", pools="20,60",
                sessions="midday,evening,pooled", choice="prev_block_rank", **over):
    d = {"--dataset": fx["dataset"], "--dataset-sha256": sha(fx["dataset"]), "--npz": fx["npz"],
         "--generation-digest": fx["digest"], "--scorer-config": fx["config"], **FX_FRAME,
         "--origins": origins, "--grade-sessions": sessions, "--model-choice": choice,
         "--pool-sizes": pools, "--families": families, "--trials": "2", "--k-folds": "3",
         "--step5-cuda": "hidden", "--step3-workers": "2", "--step3-chunk-size": "40",
         "--vote-harness": C.vote, "--vote-harness-sha256": C.vsha, "--grader": C.grader, "--grader-sha256": C.gsha,
         "--vote-config-sha256": C.csha, "--vote-workers": "2", "--grader-workers": "4",
         "--control-reps": "50", "--out-dir": out}
    d.update(over)
    argv = []
    for k, v in d.items():
        if v is None:
            continue
        argv += [k] if v is True else [k, str(v)]
    return argv


def run_driver(argv, mutant=None, shim_mutant=None):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env.pop("E2E_WF_MUTANT", None)
    env.pop("E2E_SHIM_MUTANT", None)
    if mutant:
        env["E2E_WF_MUTANT"] = mutant
    if shim_mutant:
        env["E2E_SHIM_MUTANT"] = shim_mutant
    t0 = time.time()
    p = subprocess.run([PY, "-B", str(DRIVER)] + [str(x) for x in argv], capture_output=True,
                       text=True, env=env)
    p.seconds = round(time.time() - t0, 1)
    return p


def perturbed_fixture(fx, d: Path, cut: int, seed=7):
    """Copy of the fixture whose draws at combined index >= cut are replaced."""
    d.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    ds = json.load(open(fx["dataset"]))
    for t in range(cut, len(ds)):
        ds[t]["draw"] = int((ds[t]["draw"] + 1 + rng.integers(0, 998)) % 1000)
    (d / "daily3.json").write_text(json.dumps(ds, indent=2))
    hold = [r["draw"] for r in ds[fx["n_train"]:]]
    (d / "holdout_history.json").write_text(json.dumps(hold, indent=2))
    for n in ("train_history.json", "optimal_scorer_config.json"):
        shutil.copyfile(Path(fx["dir"]) / n, d / n)
    shutil.copyfile(fx["npz"], d / "survivors.npz")
    return {"dir": d, "npz": d / "survivors.npz", "digest": fx["digest"], "train": d / "train_history.json",
            "hold": d / "holdout_history.json", "dataset": d / "daily3.json",
            "config": d / "optimal_scorer_config.json", "n_train": fx["n_train"]}


def fp_step3(out: Path, i: int):
    o = out / f"origin_{i}"
    s3 = ST3.strip_ts(json.load(open(o / "step3" / "survivors_with_scores.json")))  # metadata.timestamp = run time
    return {"label_block": sha(o / "copy" / "label_block_history.json"),
            "train_history": sha(o / "copy" / "train_history.json"),
            "step3_survivors": hashlib.sha256(json.dumps(s3, sort_keys=True).encode()).hexdigest()}


def fingerprint(out: Path, i: int):
    """Everything upstream of the grade for origin i, plus the outcomes (separately)."""
    o = out / f"origin_{i}"
    rec = json.load(open(o / "origin_summary.json"))
    fp = fp_step3(out, i)
    for key, w in rec["step5"].items():
        fp[f"winner/{key}"] = (w["model_type"], w["checkpoint_sha256"], w["cv_score"], w["rule_used"],
                               json.dumps(w["choice_metric"], sort_keys=True))
        for fam, cv in w["cv_ranking"]:
            fp[f"cv/{key}/{fam}"] = cv
    for key, v in rec["leak_checks"]["L3_step5_inputs"].items():
        fp[f"step5_lottery/{key}"] = sha(v["lottery_data"])
    for fam_dir in sorted((o / "step5").glob("s*/*/models/reinforcement")):
        for f in sorted(fam_dir.glob("best_model.*")):
            if not f.name.endswith(".json"):
                fp[f"checkpoint/{fam_dir.parts[-4]}/{fam_dir.parts[-3]}"] = sha(f)
    fp["vote_history"] = rec["leak_checks"]["L5_vote_seal"]["history_sha256"]
    fp["vote_seal"] = rec["leak_checks"]["L5_vote_seal"]["seal_from"]
    for v in rec["votes"]:
        pools = [(json.loads(l)["target"]["combined_index"], json.loads(l)["pool"])
                 for l in open(Path(v["dir"]) / "pools.jsonl")]
        fp[f"pools/{v['arm_id']}"] = hashlib.sha256(json.dumps(pools).encode()).hexdigest()
    outcomes = [(json.loads(l)["combined_index"], json.loads(l).get("drawn"))
                for l in open(o / "per_draw_hits.jsonl")]
    return fp, outcomes


def diff_keys(a, b):
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def all_json_label(out: Path):
    bad = []
    for p in out.rglob("*.json"):
        rel = p.relative_to(out).parts
        if any(x in ("copy", "step3", "step5", "vote", "grade", "frozen", "split") for x in rel[1:2]) and not p.name.endswith("DONE.json"):
            continue           # files written by the called tools; their stamps are checked separately
        if p.name.endswith(".partial") or "_env" in rel or (len(rel) > 1 and rel[1] == "winner"
                                                             and p.name != "WINNER.json"):
            continue
        try:
            obj = json.load(open(p))
        except Exception:
            bad.append(str(p))
            continue
        if isinstance(obj, dict) and obj.get("label") != "DEVELOPMENT":
            bad.append(str(p))
    return bad


CAPS_FILES = ["prediction_generator.py", "full_scoring_worker.py", "survivor_scorer.py",
              "meta_prediction_optimizer_anti_overfit.py", "models/feature_manifest.py"]


def w10_switches(C, fx, base, A, w1):
    caps = load(HERE / "e2e_pipeline_caps.py", "e2e_pipeline_caps")
    vc = json.load(open(VCFG))
    (vc.get("prediction", vc))["schema_check"] = "fail"
    vc1 = base / "w10_vote_config_ll1.json"
    vc1.write_text(json.dumps(vc, indent=2))
    REQ = {"LL-1": ({"schema_check": "fail"}, {"--vote-config": str(vc1), "--vote-config-sha256": sha(vc1)}),
           "LL-2": ({"--per-seed-sieve-features": "on"}, {"--step3-extra-args": "--per-seed-sieve-features on"}),
           "LL-3": ({"--exclude-feature-groups": "dead"},
                    {"--step5-extra-args": "--enable-diagnostics --exclude-feature-groups dead"})}
    # the "without" side: this tree's committed pipeline files at --caps-without-rev
    wo = base / "w10_without"
    for f in CAPS_FILES:
        g = subprocess.run(["git", "-C", str(WT), "show", f"{A.caps_without_rev}:{f}"], capture_output=True)
        if g.returncode == 0:
            (wo / f).parent.mkdir(parents=True, exist_ok=True)
            (wo / f).write_bytes(g.stdout)
    wi = Path(A.caps_with_root)
    cw, cwi, here = caps.capabilities(wo), caps.capabilities(wi), caps.capabilities(WT)
    for sw, (req, over) in REQ.items():
        gate(f"W10 precondition: {A.caps_without_rev} lacks {sw} and {wi} has it",
             not cw[sw]["present"] and cwi[sw]["present"],
             f"without lacks {cw[sw]['lacks'][:1]}; with has it: {cwi[sw]['present']}")
        try:
            caps.check({sw: req}, {k: wo for k in REQ})
            ok, det = False, "no refusal"
        except caps.MissingSwitch as e:
            ok = sw in str(e) and any(str(wo / f) in str(e) for f in CAPS_FILES)
            det = str(e)[:160]
        gate(f"W10 {sw} missing: the check refuses, naming the switch and the file", ok, det)
        try:
            r = caps.check({sw: req}, {k: wi for k in REQ})
            ok, det = r[sw]["pipeline_has_switch"] and r[sw]["applied"] == req, f"applied {r[sw]['applied']}"
        except caps.MissingSwitch as e:
            ok, det = False, str(e)[:160]
        gate(f"W10 {sw} non-vacuity: the same request passes on {wi.name}'s files and is recorded applied", ok, det)
        out = base / f"w10_driver_{sw}"
        if not here[sw]["present"]:
            p = run_driver(driver_args(C, fx, out, origins="1", families="lightgbm", **over))
            ok = (p.returncode == 2 and "pipeline switch missing" in p.stderr and sw in p.stderr
                  and not out.exists())
            gate(f"W10 {sw} absent from this tree: the driver refuses before any write (exit 2, no out-dir)",
                 ok, f"rc {p.returncode}: {p.stderr.strip()[-160:]}")
        else:
            p = run_driver(driver_args(C, fx, out, origins="1", families="lightgbm", **over) + ["--plan-only"])
            rr = json.load(open(out / "run_record.json")) if (out / "run_record.json").exists() else {}
            ok = p.returncode == 0 and rr.get("pipeline_switches", {}).get(sw, {}).get("applied") == req
            gate(f"W10 {sw} present in this tree: the driver plans it and records it applied",
                 ok, f"rc {p.returncode}: {rr.get('pipeline_switches', {}).get(sw)}")
    if not here["LL-2"]["present"]:
        out = base / "w10_harness_ll2"
        copy = w1 / "origin_1" / "copy"
        p = ST3.run_harness(copy, out, "--per-seed-sieve-features", "on")
        ok = p.returncode == 2 and "LL-2 switch" in p.stderr and not out.exists()
        gate("W10 LL-2 absent: the Step 3 harness run directly refuses before its out-dir exists",
             ok, f"rc {p.returncode}: {p.stderr.strip()[-160:]}")
    else:
        print("INFO  W10 this tree has LL-2: the direct-harness refusal case does not apply", flush=True)


RUNH_INPUTS = Path("~/skipsw_run_thr3030").expanduser()
WRAPPERS = ["00_base", "01_sieve_on", "02_target_hits", "03_rankers"]
STUB_PARSER = r"""
import ast, json, os, sys, importlib.util
M = os.environ["E2E_STUB_MODULE"]; OUT = os.environ["E2E_STUB_OUT"]
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("m5_for_stub", M)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
fn = next(n for n in ast.parse(open(M).read()).body if isinstance(n, ast.FunctionDef) and n.name == "main")
body = []
for st in fn.body:
    body.append(st)
    if isinstance(st, ast.Assign) and isinstance(st.value, ast.Call) and getattr(st.value.func, "attr", "") == "parse_args":
        break
else:
    raise SystemExit("stub: no parse_args in main()")
body += ast.parse("__stub_out__ = vars(args)").body
mod = ast.Module(body=body, type_ignores=[]); ast.fix_missing_locations(mod)
g = dict(vars(m)); exec(compile(mod, M, "exec"), g)
a = g["__stub_out__"]
rows = json.load(open(a["survivors"]))
json.dump({"args": a, "holdout_quality_read": sorted({r.get("holdout_quality") for r in rows})},
          open(OUT, "w"), default=str)
"""


def wrapper_tree(dst: Path, fsw: bytes, ss: bytes) -> Path:
    """A tree like this one, with the given Step 3 pipeline files: scripts/diagnostics and the run H
    wrappers are real copies (the driver resolves its own tree from __file__); every other top-level
    entry is a read-only symlink into this tree."""
    (dst / "scripts" / "diagnostics").mkdir(parents=True)
    for f in HERE.iterdir():
        if f.is_file() and f.suffix == ".py":
            shutil.copyfile(f, dst / "scripts" / "diagnostics" / f.name)
    rc = dst / "docs" / "drafts" / "loop_v1" / "runH_commands"
    rc.mkdir(parents=True)
    for w in WRAPPERS:
        shutil.copyfile(WT / "docs/drafts/loop_v1/runH_commands" / f"{w}.sh", rc / f"{w}.sh")
    for e in WT.iterdir():
        if e.name in ("scripts", "docs", "data", "logs", ".git", "tests", "__pycache__",
                      "full_scoring_worker.py", "survivor_scorer.py"):
            continue
        os.symlink(e, dst / e.name)
    (dst / "full_scoring_worker.py").write_bytes(fsw)
    (dst / "survivor_scorer.py").write_bytes(ss)
    (dst / "data").mkdir()
    return dst


def run_wrapper(tree: Path, name: str, stub_dir: Path):
    mark = stub_dir / f"taskset_called_{tree.name}_{name}"
    (stub_dir / "taskset").write_text("#!/bin/bash\n" + 'echo "$@" > "$STUB_MARK"\nexit 0\n')
    (stub_dir / "taskset").chmod(0o755)
    env = dict(os.environ, CORES="0", W="1", STAGE="smoke", CUDA_VISIBLE_DEVICES="",
               PYTHONDONTWRITEBYTECODE="1", STUB_MARK=str(mark), PATH=f"{stub_dir}:{os.environ['PATH']}")
    p = subprocess.run(["bash", str(tree / "docs/drafts/loop_v1/runH_commands" / f"{name}.sh")],
                       capture_output=True, text=True, env=env, cwd=str(base_cwd()))
    if p.returncode == 0:              # the launch is backgrounded (nohup ... &): give the stub time
        t0 = time.time()
        while not mark.exists() and time.time() - t0 < 20:
            time.sleep(0.2)
    out = tree / "data" / f"e2e_runH_wf_{name}_smoke"
    files = {"log": (tree / "logs" / f"e2e_runH_wf_{name}_smoke.log").exists(),
             "logs_dir": (tree / "logs").exists(),
             "time": Path(str(out) + ".time").exists(), "out_dir": out.exists(), "launched": mark.exists()}
    return p, files


def base_cwd():
    return Path("/")


def w11_wrappers(base, A):
    if not (RUNH_INPUTS / "daily3.json").is_file():
        gate("W11 precondition: run H inputs present (read only)", False, f"{RUNH_INPUTS} missing")
        return
    stub = base / "w11_stub"
    stub.mkdir()
    get = lambda f: subprocess.run(["git", "-C", str(WT), "show", f"{A.caps_without_rev}:{f}"],
                                   capture_output=True).stdout
    wo = wrapper_tree(base / "w11_without", get("full_scoring_worker.py"), get("survivor_scorer.py"))
    wi_root = Path(A.caps_with_root)
    wi = wrapper_tree(base / "w11_with", (wi_root / "full_scoring_worker.py").read_bytes(),
                      (wi_root / "survivor_scorer.py").read_bytes())
    caps = load(HERE / "e2e_pipeline_caps.py", "e2e_pipeline_caps_w11")
    gate("W11 precondition: the without-tree lacks LL-2 and the with-tree has it",
         not caps.capabilities(wo)["LL-2"]["present"] and caps.capabilities(wi)["LL-2"]["present"], "")
    p, f = run_wrapper(wo, "01_sieve_on", stub)
    txt = p.stdout + p.stderr
    ok = (p.returncode != 0 and "PREFLIGHT REFUSED" in txt and "LL-2" in txt and "full_scoring_worker.py" in txt
          and not any(f.values()))
    gate("W11 01_sieve_on without LL-2: non-zero exit; no log, no .time, no out-dir, no logs/, no launch",
         ok, f"rc {p.returncode} {f} :: {txt.strip()[-220:]}")
    p, f = run_wrapper(wi, "01_sieve_on", stub)
    ok = p.returncode == 0 and "started" in p.stdout and f["launched"] and f["log"] and not f["out_dir"]
    gate("W11 non-vacuity: 01_sieve_on with LL-2 passes the same preflight and reaches the launch",
         ok, f"rc {p.returncode} {f} :: {(p.stdout + p.stderr).strip()[-220:]}")
    for w in ("00_base", "02_target_hits", "03_rankers"):
        p, f = run_wrapper(wo, w, stub)
        ok = p.returncode == 0 and "started" in p.stdout and f["launched"] and not f["out_dir"]
        gate(f"W11 {w} on the files without LL-2: preflight passes, launch reached",
             ok, f"rc {p.returncode} {f} :: {(p.stdout + p.stderr).strip()[-200:]}")


def w12_passthroughs(C, fx, base, w1):
    def refused(tag, over, want, mutant=None, check_only=True):
        out = base / f"w12_{tag}"
        argv = driver_args(C, fx, out, origins="1", families="lightgbm", **over) + (["--check-only"] if check_only else [])
        p = run_driver(argv, mutant=mutant)
        return (p.returncode == 2 and "passthrough refused" in p.stderr and want in p.stderr
                and not out.exists()), p, out
    cases = [
        ("abbrev_s3", {"--step3-extra-args": "--per-seed-sieve-f on"}, "abbreviation", "passthrough_abbrev_allowed"),
        ("surv_eq", {"--step5-extra-args": "--enable-diagnostics --survivors=/nonexistent/x.json"}, "--survivors",
         "passthrough_eq_form_ignored"),
        ("folds_sp", {"--step5-extra-args": "--enable-diagnostics --k-folds 2"}, "--k-folds",
         "passthrough_driver_set_ignored"),
        ("folds_eq", {"--step5-extra-args": "--enable-diagnostics --k-folds=2"}, "--k-folds", None),
        ("trials_nn", {"--nn-args": "--normalize-features --trials=3"}, "--trials", None),
        ("outdir_s3", {"--step3-extra-args": "--out-dir data/elsewhere"}, "--out-dir", None),
        ("unknown_s5", {"--step5-extra-args": "--enable-diagnostics --no-such-option"}, "not an option", None),
    ]
    for tag, over, want, mname in cases:
        ok, p, out = refused(tag, over, want)
        gate(f"W12 driver refuses {list(over.values())[0]!r} before any write (exit 2, names it)",
             ok, f"rc {p.returncode}: {p.stderr.strip()[-170:]}")
        if mname:
            okm, pm, _ = refused(tag + "_mut", over, want, mutant=mname)
            gate(f"W12 mutant {mname}: applied, ran, the refusal gate catches it (not refused under the mutant)",
                 not okm, f"rc {pm.returncode}: {(pm.stdout + pm.stderr).strip()[-140:]}")
    out = base / "w12_checkonly_ok"
    p = run_driver(driver_args(C, fx, out, origins="1", families="lightgbm") + ["--check-only"])
    gate("W12 non-vacuity: default passthroughs pass --check-only, which writes nothing",
         p.returncode == 0 and "PREFLIGHT OK" in p.stdout and not out.exists(), f"rc {p.returncode}: {p.stdout.strip()[-150:]}")
    # parsers a passthrough reaches: no abbreviations
    copy = w1 / "origin_1" / "copy"
    o3 = base / "w12_harness_abbrev"
    ph = ST3.run_harness(copy, o3, "--per-seed-sieve-f", "on")
    gate("W12 the Step 3 harness parser rejects --per-seed-sieve-f (allow_abbrev=False), nothing created",
         ph.returncode == 2 and "unrecognized arguments" in ph.stderr and not o3.exists(), f"rc {ph.returncode}: {ph.stderr.strip()[-120:]}")
    cap0 = base / "w12_shim_abbrev_cap"
    ps = subprocess.run([PY, "-B", str(SHIM), "--split-seed", "1", "--capture-dir", str(cap0), "--script", str(MODULE5),
                         "--target", "holdout_hits", "--", "--survivors", "x"], capture_output=True, text=True,
                        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    gate("W12 the shim parser rejects --target (abbreviation of --target-field), nothing created",
         ps.returncode == 2 and not cap0.exists(), f"rc {ps.returncode}: {ps.stderr.strip()[-120:]}")
    # Beta's regression gate: what the optimizer actually reads, through the real shim and main's parser body
    d = base / "w12_regress"; d.mkdir()
    src = d / "survivors_src.json"
    src.write_text(json.dumps([{"seed": i, "features": {"f1": float(i)}, "holdout_quality": 0.045, "holdout_hits": 2}
                               for i in range(6)]))
    stub = d / "optimizer_parser_stub.py"; stub.write_text(STUB_PARSER)
    def shim_run(tag, extra, smut=None):
        cap, res = d / f"cap_{tag}", d / f"parsed_{tag}.json"
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
                   E2E_STUB_MODULE=str(MODULE5), E2E_STUB_OUT=str(res))
        env.pop("E2E_SHIM_MUTANT", None)
        if smut:
            env["E2E_SHIM_MUTANT"] = smut
        q = subprocess.run([PY, "-B", str(SHIM), "--split-seed", "1", "--capture-dir", str(cap), "--script", str(stub),
                            "--target-field", "holdout_hits", "--", "--survivors", str(src),
                            "--lottery-data", str(fx["train"]), "--model-type", "lightgbm", "--output-dir", str(d / "o"),
                            "--trials", "2", "--k-folds", "3"] + extra, capture_output=True, text=True, env=env)
        parsed = json.load(open(res)) if res.exists() else None
        summ = json.load(open(cap / "shim_summary.json")) if (cap / "shim_summary.json").exists() else {}
        return q, parsed, summ
    def regress_ok(q, parsed, summ):
        derived = (summ.get("target") or {}).get("derived_survivors")
        return (q.returncode == 0 and parsed is not None and parsed["args"]["survivors"] == derived
                and parsed["holdout_quality_read"] == [2] and int(parsed["args"]["k_folds"]) == 3)
    q, parsed, summ = shim_run("plain", [])
    gate("W12 regression: a holdout_hits run makes the optimizer read the derived file (target 2, not 0.045) "
         "with the declared 3 folds", regress_ok(q, parsed, summ),
         f"rc {q.returncode}; parsed {parsed and {k: parsed['args'].get(k) for k in ('survivors', 'k_folds')}} "
         f"read {parsed and parsed['holdout_quality_read']} {q.stderr.strip()[-150:]}")
    q, parsed, summ = shim_run("eq", [f"--survivors={src}"])
    gate("W12 regression: a trailing --survivors=SRC is refused by the shim before the optimizer runs",
         q.returncode != 0 and parsed is None and "refuse" in (q.stdout + q.stderr),
         f"rc {q.returncode}: {(q.stdout + q.stderr).strip()[-150:]}")
    q, parsed, summ = shim_run("eq_mut", [f"--survivors={src}"], smut="survivors_eq_allowed")
    killed = not (q.returncode != 0 and parsed is None) and not regress_ok(q, parsed, summ)
    gate("W12 mutant survivors_eq_allowed: applied, ran, killed (the optimizer reads the source file, "
         "target 0.045)", killed and parsed is not None and parsed["holdout_quality_read"] == [0.045],
         f"rc {q.returncode}; read {parsed and parsed['holdout_quality_read']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps-without-rev", default="HEAD",
                    help="W10: git revision of this tree whose pipeline files lack the LL switches")
    ap.add_argument("--caps-with-root", default=str(Path("~/tfm_worktrees/e2e-map").expanduser()),
                    help="W10 non-vacuity: a tree whose pipeline files have the LL switches")
    A = ap.parse_args()
    t_all = time.time()
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = WT / "data" / f"e2e_wf_selftest_{stamp}"
    base.mkdir(parents=True)
    C = Ctx()
    # The grader worktree is being edited by its own agent: run a sha-verified snapshot copy
    # (it imports nothing from its repo; provenance is off). The vote harness imports its own
    # repo, so it runs in place, pinned.
    (base / "pins").mkdir()
    shutil.copyfile(GRADER, base / "pins" / "e2e_grader.py")
    C.grader = str(base / "pins" / "e2e_grader.py")
    C.vote = str(snapshot_vote(base / "pins" / "vote"))
    C.vsha, C.gsha, C.csha = sha(C.vote), sha(C.grader), sha(VCFG)
    C.gsrc, C.vsrc = sha(GRADER), sha(VOTE)
    print(f"selftest dir {base}; pins vote {C.vsha} grader {C.gsha} (snapshot of {GRADER}) "
          f"config {C.csha}", flush=True)
    before = {r: (git_status(r), pycaches(r)) for r in (MAIN, VOTE.parents[2], GRADER.parents[2])}

    fx = ST3.make_fixture(base / "fx", rows=80, n_train=200, n_hold=160)
    idx, _ = WF.load_dataset_index(Path(fx["dataset"]))

    # ---------------------------------------------------------------- W1
    w1 = base / "w1"
    a1 = driver_args(C, fx, w1)
    p = run_driver(a1)
    (base / "w1.stdout").write_text(p.stdout + p.stderr)
    ok = p.returncode == 0
    gate("W1 rc 0, origins 1-3, model choice prev_block_rank", ok, f"rc {p.returncode}, {p.seconds} s")
    if not ok:
        print(p.stdout[-3000:], p.stderr[-3000:])
        return 1
    decl = json.load(open(w1 / "run_record.json"))
    bt = decl["frame"]["block_table"]
    cuts = [b["start_index"] for b in bt] + [bt[-1]["end_index"]]
    tab = json.load(open(w1 / "table.json"))
    # per origin: (A1 + A0) x 2 pools + 2 family arms at 60 = 6 votes, x 3 session views
    gate("W1 table rows = 3 origins x 6 votes x 3 session views", len(tab["rows"]) == 54, f"{len(tab['rows'])}")
    bad = all_json_label(w1)
    pd_rows = [json.loads(l) for i in (1, 2, 3) for l in open(w1 / f"origin_{i}" / "per_draw_hits.jsonl")]
    gate("W1 DEVELOPMENT on every driver json and per-draw row",
         not bad and all(r["label"] == "DEVELOPMENT" and r["mode"] == "DEVELOPMENT" for r in pd_rows)
         and "DEVELOPMENT" in (w1 / "TABLE.md").read_text().splitlines()[0],
         f"bad {bad[:3]}, rows {len(pd_rows)}")
    gdirs = [d for d in w1.glob("origin_*/grade/*") if d.is_dir()]
    gate("W1 grader DEVELOPMENT marker in every grade dir",
         len(gdirs) == 54 and all((d / "DEVELOPMENT").is_file() for d in gdirs), f"{len(gdirs)} dirs")
    recs = [json.load(open(w1 / f"origin_{i}" / "origin_summary.json")) for i in (1, 2, 3)]

    def lc_ok(r):
        lc = r["leak_checks"]
        return (lc["L1_label_block"]["ok"] and lc["L2_step3_inputs"]["ok"]
                and all(v["ok"] for v in lc["L3_step5_inputs"].values())
                and all(v["ok"] for v in lc["L4_choice"].values())
                and lc["L5_vote_seal"]["ok"] and lc["L6_grade_block"]["ok"])
    gate("W1 all runtime leak checks ok (L1-L6)", all(lc_ok(r) for r in recs))
    exp = [[[0, cuts[i] - 40], [cuts[i] - 40, cuts[i]]] for i in (1, 2, 3)]
    got = [[r["train"], r["label_block"]] for r in recs]
    tr_len = [len(json.load(open(w1 / f"origin_{i}" / "copy" / "train_history.json"))) for i in (1, 2, 3)]
    gate("W1 train [0, b_i - L) and labels [b_i - L, b_i): the train end moves with the origin",
         got == exp and tr_len == [cuts[i] - 40 for i in (1, 2, 3)], f"{got}; train lengths {tr_len}")
    gate("W1 range starts at the earliest workable origin; cuts fall on date changes",
         decl["frame"]["range"][0] == decl["frame"]["earliest_workable_origin"] == 140
         and all(idx[c - 1][0] != idx[c][0] for c in cuts[1:-1]), f"range {decl['frame']['range']}, cuts {cuts}")
    rules = [r["step5"]["s42"]["rule_used"] for r in recs]
    ev_ok = all(all(e[1] <= r["block"]["start_index"] for e in r["step5"]["s42"]["evidence_blocks"]) for r in recs)
    ev_prev = all(r["step5"]["s42"]["evidence_blocks"] and r["step5"]["s42"]["evidence_blocks"][0][:2] ==
                  [recs[n]["block"]["start_index"], recs[n]["block"]["end_index"]] for n, r in enumerate(recs[1:]))
    gate("W1 choice: origin 1 falls back to cv; origins 2-3 use the previous origin's graded block",
         rules == ["cv", "prev_block_rank", "prev_block_rank"] and ev_ok and ev_prev, f"rules {rules}")
    order_ok, ro_ok = True, True
    for i in (1, 2, 3):
        wd = w1 / f"origin_{i}" / "winner" / "s42"
        wj = json.load(open(wd / "WINNER.json"))
        first_vote = min(json.load(open(f))["started"] for f in (w1 / f"origin_{i}" / "vote").glob("*.DONE.json"))
        order_ok &= _dt.datetime.fromisoformat(wj["frozen_at"]).replace(tzinfo=None) <= _dt.datetime.fromisoformat(first_vote)
        ro_ok &= not os.access(wd / "WINNER.json", os.W_OK) and not (wd.stat().st_mode & 0o222)
        for fd in (w1 / f"origin_{i}" / "frozen" / "s42").iterdir():
            ro_ok &= not (fd.stat().st_mode & 0o222)
    gate("W1 winner and every family frozen read-only before the first vote", order_ok and ro_ok)
    cnt_ok, pooled_ok = True, True
    for r in recs:
        by = {}
        for g in r["grades"]:
            want = sum(r["block"]["sessions"].values()) if g["session"] == "pooled" else r["block"]["sessions"].get(g["session"], 0)
            cnt_ok &= g["n_graded"] == want
            by.setdefault(g["arm_id"], {})[g["session"]] = g
        for arm, d in by.items():
            pooled_ok &= d["pooled"]["hits"] == d["midday"]["hits"] + d["evening"]["hits"]
            pooled_ok &= abs(d["pooled"]["expected_hits"] - d["midday"]["expected_hits"] - d["evening"]["expected_hits"]) < 1e-9
    gate("W1 graded draws = block rows per session view; pooled = midday + evening", cnt_ok and pooled_ok)
    rk_ok = True
    for r in recs:
        for g in r["grades"]:
            if g["pool_size"] == 60:
                det = [json.loads(l) for l in open(Path(g["dir"]) / "per_draw_detail.jsonl")]
                pools = {json.loads(l)["target"]["combined_index"]: json.loads(l)["pool"]
                         for l in open(w1 / f"origin_{r['origin']}" / "vote" / g["arm_id"] / "pools.jsonl")}
                rs = []
                for row in det:
                    if row["status"] != "graded":
                        continue
                    order = list(dict.fromkeys(pools[row["combined_index"]]))
                    rs.append(order.index(row["drawn"]) + 1 if row["drawn"] in order else (len(order) + 1001) / 2)
                    rk_ok &= (row["drawn"] in order) == bool(row["hit"])
                rk_ok &= g["rank"] is not None and abs(g["rank"]["mean_rank"] - sum(rs) / len(rs)) < 1e-9
    gate("W1 rank: mean rank re-derived independently from pools + grader outcomes; hit <=> ranked", rk_ok)
    rg_ok = all(g["rank_grader"] is not None and g["rank_check"].startswith("driver == grader")
                and (g["rank"] is None or abs(g["rank_grader"]["mean_rank"] - g["rank"]["mean_rank"]) < 1e-9)
                for r in recs for g in r["grades"])
    gate("W9 grader R24 rank summary on every grade; equals the driver's rank (ordering off)", rg_ok)
    sw_ok, sw_err = True, ""
    for i in range(1, decl["frame"]["blocks"]):
        w = WF.selfplay_window(decl["frame"], i)
        sw_ok &= (w["validate_block"][1] == w["use_block"][0] == w["seal"] and w["validate_labels"][1] == w["validate_block"][0]
                  and w["validate_train"] == [0, w["validate_labels"][0]] and w["interface"] == WF.SELFPLAY_INTERFACE
                  and w["validate_labels"][0] == w["validate_block"][0] - 40)
    for bad in (0, decl["frame"]["blocks"]):
        try:
            WF.selfplay_window(decl["frame"], bad)
            sw_ok, sw_err = False, f"no error for {bad}"
        except ValueError:
            pass
    gate("W9 selfplay_window interface: validate block i-1 with labels before it, use block i, seal b_i", sw_ok, sw_err)
    w1_table_sha = sha(w1 / "table.json")

    # ---------------------------------------------------------------- W2 leak
    cut = cuts[2]
    fxB = perturbed_fixture(fx, base / "fxB", cut)
    w2 = base / "w2B"
    p = run_driver(driver_args(C, fxB, w2, origins="1,2"))
    gate("W2 perturbed run (draws >= b_2 replaced) rc 0", p.returncode == 0, f"rc {p.returncode}, {p.seconds} s")
    if p.returncode == 0:
        fa, oa = fingerprint(w1, 2)
        fb, ob = fingerprint(w2, 2)
        d = diff_keys(fa, fb)
        gate("W2 LEAK: labels, train, Step 3, Step 5 inputs, CV, checkpoints, choice, vote history and "
             "every pool identical under perturbed future draws", not d, f"{len(fa)} fields compared; differing {d}")
        gate("W2 perturbation effective: outcomes of block 2 differ", oa != ob and len(oa) == len(ob),
             f"{sum(1 for x, y in zip(oa, ob) if x != y)} of {len(oa)} rows differ")
        f1a, _ = fingerprint(w1, 1)
        f1b, _ = fingerprint(w2, 1)
        gate("W2 origin 1 (all before the cut) identical too", not diff_keys(f1a, f1b), f"{diff_keys(f1a, f1b)}")

    expect = {"label_includes_origin": "step3_survivors", "seal_after_origin": "vote_history",
              "step5_lottery_future": "step5_lottery/s42/lightgbm"}
    for m, field in expect.items():
        oa_, ob_ = base / f"w2m_{m}_A", base / f"w2m_{m}_B"
        kw = dict(origins="2", families="lightgbm", pools="20", sessions="midday", choice="cv")
        pa = run_driver(driver_args(C, fx, oa_, **kw), mutant=m + "+nocheck")
        pb = run_driver(driver_args(C, fxB, ob_, **kw), mutant=m + "+nocheck")
        applied = (oa_ / "run_record.json").exists() and \
            json.load(open(oa_ / "run_record.json"))["mutant"] == m + "+nocheck"
        try:
            if (oa_ / "origin_2" / "origin_summary.json").exists() and (ob_ / "origin_2" / "origin_summary.json").exists():
                d = diff_keys(fingerprint(oa_, 2)[0], fingerprint(ob_, 2)[0])
            else:
                d = diff_keys(fp_step3(oa_, 2), fp_step3(ob_, 2))
        except Exception as e:
            d = [f"error {e!r}"]
        gate(f"W2m {m}+nocheck: applied, ran, the leak gate sees it ({field})", applied and field in d,
             f"rc {pa.returncode}/{pb.returncode}; differing {d}")

    # ---------------------------------------------------------------- W3 runtime checks
    w3 = {"label_includes_origin": ("1", 4, "L1 origin 1"), "step5_lottery_future": ("1", 4, "L3 origin 1"),
          "choice_uses_current_block": ("1,2", 4, "L4 origin 2"),
          "grade_before_freeze": ("1", 4, "L4 origin 1"), "seal_after_origin": ("1", 4, "L5 origin 1"),
          "grade_previous_block": ("1", 4, "L6 origin 1"), "counted_mode": ("1", 3, "o1/grade/"),
          "table_drop_origin": ("1", 3, "table incomplete"),
          "rank_off_by_one": ("1", 3, "ranks differ")}
    for m, (orig, rc, msg) in w3.items():
        out = base / f"w3_{m}"
        p = run_driver(driver_args(C, fx, out, origins=orig, families="lightgbm", pools="20",
                                   choice="prev_block_hits" if m == "choice_uses_current_block" else "cv",
                                   **({"--choice-pool-size": "20"} if m == "choice_uses_current_block" else {})),
                       mutant=m)
        applied = (out / "run_record.json").exists() and json.load(open(out / "run_record.json"))["mutant"] == m
        ok = applied and p.returncode == rc and msg in p.stdout
        extra = ""
        if m == "counted_mode":
            glog = out / "origin_1" / "grade" / "A1s42_o1_p20_midday.log"
            txt = glog.read_text() if glog.exists() else ""
            ok = ok and "GRADER ERROR" in txt and "development" in txt.lower()
            extra = txt.strip().splitlines()[-1][:200] if txt else "no grade log"
        if m == "label_includes_origin" and (out / "origin_1" / "copy" / "run_record.json").exists():
            extra = "label_count " + str(json.load(open(out / "origin_1" / "copy" / "run_record.json"))
                                         ["declared_parameters"]["label_count"])
        last = p.stdout.strip().splitlines()[-1][:180] if p.stdout.strip() else p.stderr.strip()[-180:]
        gate(f"W3 {m}: applied, ran, killed by its credited check (rc {rc}, '{msg}')", ok,
             f"rc {p.returncode}; {extra} | {last}")

    # ---------------------------------------------------------------- W4 refusals
    def refuse(name, argv, must_not_exist=True):
        p = run_driver(argv)
        out = Path(argv[argv.index("--out-dir") + 1])
        ok = p.returncode == 2 and "REFUSED" in p.stderr and (not must_not_exist or not out.exists())
        gate(f"W4 refuse: {name}", ok, f"rc {p.returncode}: {p.stderr.strip()[-150:]}")
    refuse("out-dir exists", driver_args(C, fx, w1), must_not_exist=False)
    refuse("out-dir outside data/", driver_args(C, fx, str(WT / "docs" / "wf_x")))
    refuse("wrong vote sha", driver_args(C, fx, base / "r1", **{"--vote-harness-sha256": "0" * 64}))
    refuse("wrong grader sha", driver_args(C, fx, base / "r2", **{"--grader-sha256": "0" * 64}))
    refuse("wrong config sha", driver_args(C, fx, base / "r3", **{"--vote-config-sha256": "0" * 64}))
    refuse("wrong dataset sha", driver_args(C, fx, base / "r4", **{"--dataset-sha256": "0" * 64}))
    refuse("range start before the earliest workable origin", driver_args(C, fx, base / "r5", **{"--range-start": "120"}))
    refuse("label window overlapping the sieve window",
           driver_args(C, fx, base / "r6", **{"--range-start": "140", "--sieve-window-end-index": "120"}))
    refuse("too many blocks", driver_args(C, fx, base / "r7", **{"--blocks": "200"}))
    refuse("origin out of range", driver_args(C, fx, base / "r8", origins="99"))
    refuse("unknown family", driver_args(C, fx, base / "r9", families="lightgbm,svm"))
    refuse("neural_net family arms without --vote-allow-gpu", driver_args(C, fx, base / "r10", families="neural_net,lightgbm"))
    refuse("choice pool size not voted", driver_args(C, fx, base / "r11", **{"--choice-pool-size": "33"}))
    refuse("unknown grade session", driver_args(C, fx, base / "r12", sessions="midday,night"))
    refuse("resume with a changed declaration",
           driver_args(C, fx, w1, pools="20", **{"--resume": True}), must_not_exist=False)

    # ---------------------------------------------------------------- W5 resume
    p = run_driver(a1 + ["--resume"])
    n_skip = p.stdout.count("skip (resume, verified)")
    n_start = sum(1 for l in p.stdout.splitlines() if "] start " in l)
    n_done = len(list(w1.rglob("*.DONE.json")))
    gate("W5 resume skips every step after verifying it, same table", p.returncode == 0 and
         n_start == 0 and n_skip == n_done and sha(w1 / "table.json") == w1_table_sha,
         f"rc {p.returncode}, skipped {n_skip} of {n_done}, started {n_start}")

    # ---------------------------------------------------------------- W6 target
    sh = [json.load(open(f)) for f in w1.glob("origin_*/step5/s42/*/shim_capture/shim_summary.json")]
    gate("W6 default target: no rewrite, module --survivors is the Step 3 output",
         len(sh) == 6 and all(s["target"]["rewritten"] is False and s["target"]["field"] == "holdout_quality"
                              and s["argv"][s["argv"].index("--survivors") + 1].endswith("step3/survivors_with_scores.json")
                              for s in sh), f"{len(sh)} shim summaries")
    src = w1 / "origin_3" / "step3" / "survivors_with_scores.json"
    rows = json.load(open(src))
    names = sorted(rows[0]["holdout_features"])
    fld = max(names, key=lambda n: len({r["holdout_features"][n] for r in rows}))
    want = np.asarray([r["holdout_features"][fld] for r in rows], dtype=np.float32)
    train3 = w1 / "origin_3" / "split" / "train_history.json"

    def shim_run(tag, field, mutant=None, survivors=src):
        rd = base / f"w6_{tag}"
        rd.mkdir()
        shutil.copyfile(WT / "train_single_trial.py", rd / "train_single_trial.py")
        env = WF.base_env(rd / "_env", "")
        env.update(PYTHONPATH=str(WT), S88_FORCE_LGBM_CPU="1")
        env.pop("E2E_SHIM_MUTANT", None)
        if mutant:
            env["E2E_SHIM_MUTANT"] = mutant
        cmd = [PY, "-u", "-B", str(SHIM), "--split-seed", "42", "--capture-dir", str(rd / "cap"),
               "--script", str(MODULE5), "--target-field", field, "--"]
        if survivors:
            cmd += ["--survivors", str(survivors)]
        cmd += ["--lottery-data", str(train3), "--model-type", "lightgbm",
                "--output-dir", str(rd / "models" / "reinforcement"), "--trials", "2", "--k-folds", "3"]
        p = subprocess.run(cmd, cwd=rd, env=env, capture_output=True, text=True)
        side = rd / "models" / "reinforcement" / "best_model.meta.json"
        sq = json.load(open(side))["signal_quality"] if side.exists() else None
        summ = json.load(open(rd / "cap" / "shim_summary.json")) if (rd / "cap" / "shim_summary.json").exists() else None
        return p, sq, summ

    def matches(sq):
        return (sq is not None and abs(sq["target_mean"] - float(want.mean())) < 1e-5
                and abs(sq["target_std"] - float(want.std())) < 1e-5
                and sq["unique_target_values"] == len(np.unique(want)))
    p, sq, summ = shim_run("hf", f"holdout_features.{fld}")
    der = json.load(open(summ["target"]["derived_survivors"])) if summ and summ["target"].get("derived_survivors") else []
    same_rest = len(der) == len(rows) and all(
        {k: v for k, v in a_.items() if k != "holdout_quality"} == {k: v for k, v in b_.items() if k != "holdout_quality"}
        for a_, b_ in zip(der, rows))
    gate(f"W6 target holdout_features.{fld}: sidecar target statistics are that field's; rows otherwise unchanged",
         p.returncode == 0 and matches(sq) and same_rest and summ["target"]["original_sha256"] == sha(src),
         f"rc {p.returncode}; sidecar mean {sq and sq['target_mean']} vs {float(want.mean()):.6f}, "
         f"unique {sq and sq['unique_target_values']} vs {len(np.unique(want))}")
    p, sq, summ = shim_run("mut", f"holdout_features.{fld}", mutant="target_not_passed")
    gate("W6 mutant target_not_passed: applied, ran, caught by the statistics check",
         p.returncode == 0 and summ["target"].get("mutant") == "target_not_passed" and not matches(sq),
         f"sidecar mean {sq and sq['target_mean']}")
    bad_rows = json.load(open(src))
    del bad_rows[3]["holdout_hits"]
    badf = base / "w6_missing.json"
    badf.write_text(json.dumps(bad_rows))
    for tag, field, surv, needle in [("ref1", "features.score", src, "not allowed"),
                                     ("ref2", "holdout_hits", badf, "row 3 holdout_hits"),
                                     ("ref3", "holdout_hits", None, "exactly one '--survivors")]:
        p, sq, summ = shim_run(tag, field, survivors=surv)
        gate(f"W6 shim refuses {tag} ({needle})", p.returncode != 0 and needle in (p.stdout + p.stderr)
             and sq is None, (p.stdout + p.stderr).strip()[-160:])

    # ---------------------------------------------------------------- W8 ranker arms (ADDENDUM_2)
    sys.path.insert(0, str(WT))
    from survivor_scorer import SurvivorScorer
    import meta_prediction_optimizer_anti_overfit as mpo
    from models.model_factory import load_model
    w8 = base / "w8"
    w8.mkdir()
    rng = np.random.default_rng(11)
    N8, T8, L8, NG = 60, 30, 60, 20
    seeds8 = [int(x) for x in rng.choice(2 ** 32 - 1, size=N8, replace=False)]
    rows8 = [{"seed": s_, "score": 0.5, "features": {"f_good": 1.0 if r < NG else 0.0,
                                                    "f_noise": float(rng.uniform())},
              "metadata": {}, "holdout_hits": 0.0, "holdout_features": {}, "holdout_quality": float(rng.uniform())}
             for r, s_ in enumerate(seeds8)]
    sc8 = SurvivorScorer(prng_type="java_lcg", mod=1000)
    V8 = np.array([sc8._generate_sequence(s_, T8 + L8, skip=0)[T8:] for s_ in seeds8])
    y8 = [int(V8[g % NG, g]) for g in range(L8)]           # planted: a good survivor's vote is drawn
    train8 = [int(v) for v in rng.integers(0, 1000, T8)]
    ds8 = [{"date": (_dt.date(2001, 1, 1) + _dt.timedelta(days=t // 2)).isoformat(),
            "session": "midday" if t % 2 == 0 else "evening", "draw": v} for t, v in enumerate(train8 + y8)]
    (w8 / "survivors.json").write_text(json.dumps(rows8))
    (w8 / "label.json").write_text(json.dumps(y8))
    (w8 / "history.json").write_text(json.dumps(train8 + y8))
    (w8 / "daily3.json").write_text(json.dumps(ds8))
    fs8 = mpo.get_feature_schema_from_data(str(w8 / "survivors.json"),
                                           ["score", "confidence", "holdout_hits", "holdout_quality"])
    tpl8 = json.load(open(next((w1 / "origin_1" / "frozen" / "s42").glob("lightgbm/best_model.meta.json"))))
    tpl8["feature_schema"] = fs8
    (w8 / "template.meta.json").write_text(json.dumps(tpl8))
    # oracle: the vote harness itself, unlearned file_order ranking, k = pool = N, every label draw
    vo = w8 / "vote_oracle"
    pv = subprocess.run([PY, "-B", C.vote, "--survivors", str(w8 / "survivors.json"), "--dataset", str(w8 / "daily3.json"),
                         "--history", str(w8 / "history.json"), "--config", str(VCFG), "--k", str(N8),
                         "--pool-size", str(N8), "--block-start", str(T8), "--block-end", str(T8 + L8),
                         "--ranking", "file_order", "--data-cutoff-index", str(T8), "--workers", "2",
                         "--out-dir", str(vo), "--arm-id", "oracle"],
                        capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    oracle = {json.loads(l)["target"]["combined_index"]: json.loads(l)["pool"] for l in open(vo / "pools.jsonl")} \
        if pv.returncode == 0 else {}

    def ranker_run(obj, tag, mutant=None):
        out = w8 / f"{tag}_{obj}"
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(WT), CUDA_VISIBLE_DEVICES="")
        env.pop("E2E_RANKER_MUTANT", None)
        if mutant:
            env["E2E_RANKER_MUTANT"] = mutant
        p = subprocess.run([PY, "-B", str(HERE / "e2e_step5_ranker.py"), "--survivors", str(w8 / "survivors.json"),
                            "--label-history", str(w8 / "label.json"), "--label-start", str(T8),
                            "--expect-label-sha256", sha(w8 / "label.json"),
                            "--sidecar-template", str(w8 / "template.meta.json"), "--pg-config", str(VCFG),
                            "--objective", obj, "--n-estimators", "60", "--max-depth", "3", "--workers", "2",
                            "--save-votes", "on", "--out-dir", str(out)], capture_output=True, text=True, env=env)
        return p, out

    def votes_equal(out):
        V = np.load(out / "votes.npy")
        return bool(oracle) and all(sorted(oracle[T8 + g]) == sorted(int(v) for v in V[:, g]) for g in range(L8)) \
            and all(oracle[T8 + g] == [int(v) for v in V[:, g]] for g in range(L8))

    def learned_good(out, fam):
        m = load_model(fam, str(next(out.glob("best_model.[jtc]*"))))
        X8 = np.array([[r["features"][n] for n in fs8["feature_names"]] for r in rows8], dtype=np.float32)
        sc = np.asarray(m.predict(X8), dtype=float)
        top = np.argsort(-sc, kind="stable")[:NG]
        return int(sum(1 for r in top if r < NG))

    gate("W8 vote oracle (vote harness, file_order, k = N on the label block) ran", pv.returncode == 0,
         pv.stderr.strip()[-200:])
    fams = {"xgb_rank_ndcg": "xgboost", "xgb_rank_pairwise": "xgboost", "lgbm_lambdarank": "lightgbm",
            "catboost_yetirank": "catboost"}
    for obj, fam in fams.items():
        p, out = ranker_run(obj, "ok")
        rr = json.load(open(out / "ranker_record.json")) if (out / "ranker_record.json").exists() else {}
        ro = rr.get("ranking_objective", {})
        ng = learned_good(out, fam) if p.returncode == 0 else -1
        gate(f"W8 {obj}: groups exact (votes = the vote harness's own pool at every label draw), "
             f"planted good survivors ranked on top", p.returncode == 0 and votes_equal(out) and ng >= NG - 2
             and ro.get("groups_total") == L8 and ro.get("items_per_group_N") == N8,
             f"rc {p.returncode}; good in top {NG}: {ng}; groups {ro.get('groups_used_train')}+{ro.get('groups_used_valid')}, "
             f"rows {ro.get('train_rows')}, valid NDCG {rr.get('training_metrics', {}).get('valid_ndcg_at_k')}")
    p, out = ranker_run("lgbm_lambdarank", "mut", mutant="relevance_shift")
    gate("W8 mutant relevance_shift: applied, ran, caught by the vote-equality gate",
         p.returncode == 0 and json.load(open(out / "ranker_record.json"))["mutant"] == "relevance_shift"
         and not votes_equal(out), f"rc {p.returncode}")
    # bounded rows: the same arm with negatives sampled
    out2 = w8 / "neg5"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(WT), CUDA_VISIBLE_DEVICES="")
    p2 = subprocess.run([PY, "-B", str(HERE / "e2e_step5_ranker.py"), "--survivors", str(w8 / "survivors.json"),
                         "--label-history", str(w8 / "label.json"), "--label-start", str(T8),
                         "--expect-label-sha256", sha(w8 / "label.json"), "--sidecar-template", str(w8 / "template.meta.json"),
                         "--pg-config", str(VCFG), "--objective", "lgbm_lambdarank", "--groups", "30",
                         "--negatives-per-group", "5", "--n-estimators", "30", "--workers", "2", "--out-dir", str(out2)],
                        capture_output=True, text=True, env=env)
    ro2 = json.load(open(out2 / "ranker_record.json"))["ranking_objective"] if p2.returncode == 0 else {}
    want_rows = None
    if p2.returncode == 0:
        pos = json.load(open(out2 / "ranker_record.json"))["positives_per_group"]
        want_rows = ro2["train_rows"]
    gate("W8 bounded size: --groups 30 --negatives-per-group 5 -> 30 groups (train+valid), rows = positives + 5 per train group",
         p2.returncode == 0 and ro2["groups_used_train"] + ro2["groups_used_valid"] == 30
         and ro2["train_rows"] == ro2["train_positives"] + 5 * ro2["groups_used_train"],
         f"rc {p2.returncode}; {ro2.get('groups_used_train')}+{ro2.get('groups_used_valid')} groups, rows {want_rows}, "
         f"positives {ro2.get('train_positives')}")
    p3 = subprocess.run([PY, "-B", str(HERE / "e2e_step5_ranker.py"), "--survivors", str(w8 / "survivors.json"),
                         "--label-history", str(w8 / "label.json"), "--label-start", str(T8),
                         "--expect-label-sha256", "0" * 64, "--sidecar-template", str(w8 / "template.meta.json"),
                         "--pg-config", str(VCFG), "--objective", "lgbm_lambdarank", "--out-dir", str(w8 / "refuse")],
                        capture_output=True, text=True, env=env)
    gate("W8 ranker refuses a label file whose sha256 is not the declared one",
         p3.returncode != 0 and "refuse" in (p3.stdout + p3.stderr) and not (w8 / "refuse").exists())
    # driver integration: two ranker arms on fixture origin 1, voted and graded
    w8d = base / "w8_driver"
    p = run_driver(driver_args(C, fx, w8d, origins="1", families="lightgbm", pools="20", sessions="midday",
                               choice="cv", **{"--ranker-arms": "lgbm_lambdarank,xgb_rank_ndcg"}))
    ok = p.returncode == 0
    if ok:
        r8 = json.load(open(w8d / "origin_1" / "origin_summary.json"))
        rr_rows = [g for g in r8["grades"] if g["role"] == "ranker"]
        ok = (len(rr_rows) == 2 and all(v["ok"] for v in r8["leak_checks"]["L7_ranker_inputs"].values())
              and all(not (Path(d["dir"]).stat().st_mode & 0o222) for d in r8["rankers"]["s42"].values())
              and {g["family"] for g in rr_rows} == {"lgbm_lambdarank", "xgb_rank_ndcg"})
    gate("W8 driver --ranker-arms: each ranker trained on [T, b_i) (L7), frozen read-only, voted and graded as its own arm",
         ok, f"rc {p.returncode}: {p.stdout.strip().splitlines()[-1][:200] if p.stdout.strip() else p.stderr[-200:]}")
    out = base / "w3_ranker_label_shift"
    p = run_driver(driver_args(C, fx, out, origins="1", families="lightgbm", pools="20", sessions="midday",
                               choice="cv", **{"--ranker-arms": "lgbm_lambdarank"}), mutant="ranker_label_shift")
    gate("W8 mutant ranker_label_shift: applied, ran, killed by L7 (rc 4)",
         p.returncode == 4 and "L7 origin 1" in p.stdout and
         json.load(open(out / "run_record.json"))["mutant"] == "ranker_label_shift",
         p.stdout.strip().splitlines()[-1][:200] if p.stdout.strip() else "")

    # ---------------------------------------------------------------- W9 full-vote ordering plumbing
    w9 = base / "w9_full_vote"
    p = run_driver(driver_args(C, fx, w9, origins="1", families="lightgbm", pools="20", sessions="midday",
                               choice="cv", **{"--vote-value-ordering": "full_vote"}))
    ok, det = p.returncode == 0, ""
    if ok:
        r9 = json.load(open(w9 / "origin_1" / "origin_summary.json"))
        pl = [json.loads(l) for l in open(w9 / "origin_1" / "vote" / "A1s42_o1_p20" / "pools.jsonl")]
        g9 = [g for g in r9["grades"] if g["pool_size"] == 20]
        ok = (all(len(x.get("value_scores") or []) == 1000 for x in pl)
              and all(g["rank_grader"] and g["rank_check"].startswith("full ordering") for g in g9))
        det = f"{len(pl)} pool records with value_scores; grader mean ranks {[round(g['rank_grader']['mean_rank'], 1) for g in g9]}, driver censored {[round(g['rank']['mean_rank'], 1) for g in g9]}"
    gate("W9 --vote-value-ordering full_vote: every pool record carries 1000 value scores; the grader ranks in them",
         ok, f"rc {p.returncode}; {det} {'' if ok else p.stdout[-300:]}")

    # ---------------------------------------------------------------- W10 pipeline switches
    w10_switches(C, fx, base, A, w1)
    # ---------------------------------------------------------------- W11, W12 (ADDENDUM_4)
    w11_wrappers(base, A)
    w12_passthroughs(C, fx, base, w1)

    # ---------------------------------------------------------------- W7 no-write
    after = {r: (git_status(r), pycaches(r)) for r in before}
    # Other agents edit the vote and grader worktrees concurrently, so their git status is
    # recorded, not gated; the main tree's status and every __pycache__ listing are gated.
    gate("W7 main tree git status unchanged; no new __pycache__ in the main tree, vote or grader worktree",
         after[MAIN] == before[MAIN] and all(after[r][1] == before[r][1] for r in before),
         f"sibling git status changed (other agents, not gated): "
         f"{[str(r) for r in before if r != MAIN and before[r][0] != after[r][0]]}")
    gate("W7 pinned snapshots unchanged (vote harness, grader, config)",
         (sha(C.vote), sha(C.grader), sha(VCFG)) == (C.vsha, C.gsha, C.csha),
         f"sources in their worktrees: vote {'unchanged' if sha(VOTE) == C.vsrc else 'CHANGED by its agent (snapshot used)'}, "
         f"grader {'unchanged' if sha(GRADER) == C.gsrc else 'CHANGED by its agent (snapshot used)'}")

    n_pass = sum(1 for _, ok in RESULTS if ok)
    summary = {"label": "DEVELOPMENT", "kind": "e2e_walk_forward_selftest", "dir": str(base),
               "pins": {"vote_snapshot": C.vsha, "vote_source_at_start": C.vsrc,
                        "grader_snapshot": C.gsha, "grader_source_at_start": C.gsrc,
                        "config": C.csha},
               "driver_sha256": sha(DRIVER), "shim_sha256": sha(SHIM), "selftest_sha256": sha(__file__),
               "gates": [{"name": n, "ok": o} for n, o in RESULTS], "passed": n_pass, "total": len(RESULTS),
               "wall_seconds": round(time.time() - t_all, 1)}
    (base / "selftest_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\n{n_pass} of {len(RESULTS)} gates pass; {summary['wall_seconds']} s; {base / 'selftest_summary.json'}")
    return 0 if n_pass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
