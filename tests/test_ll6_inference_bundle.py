#!/usr/bin/env python3
"""
tests/test_ll6_inference_bundle.py - self-test for LL-6, one inference bundle for Step 6 and
Chapter 13 (models/inference_bundle.py, prediction_generator.py, chapter_13_orchestrator.py).

Taken from ch14-learn tests/test_ll_wave2.py (the LL-6 part only; LL-7 and LL-8 land later) and
changed for landing (lead TASK_LL6_LAND_PREP_20261125):
  - every "unchanged" gate compares with a baseline read by `git show <full sha>:<file>`
    (--baseline-rev, default 7f4505a..., main before LL-6), never HEAD or a merge-base;
  - A-NORM uses the bundle's own attribution object and an independent forward pass, not the
    LL-7 signed-attribution functions;
  - Step 6's default output is compared with the baseline through the real CLI entry point
    (tests/ll6_step6_runner.py), four families, with a repeat of the baseline as the control;
  - Chapter 13's root-cause call runs with its working directory under --out (it archives to
    diagnostics_outputs/history relative to the cwd);
  - mutants are source mutations: one exact text replacement in a copy of the file, in a code
    dir put first on sys.path; each must turn its target gates red, and those gates must be green
    in the unmutated run (otherwise the mutant gets no credit).
  After Beta X-LL6-BUNDLE R1 Q5 (lead TASK_LL6_BETA_Q5_FIXES_20261125):
  - TMPDIR, XDG_CACHE_HOME, MPLCONFIGDIR and CUDA_CACHE_PATH point under --out before any library
    import, for this process and every child (pattern of scripts/diagnostics/e2e_walk_forward.py:386-394);
  - the no-write gate watches the exact boundary: this tree outside --out (git status, and a walk of
    data/ with only --out pruned), new /tmp/matplotlib-* entries, and the user's matplotlib cache and
    config dirs; a negative control writes a sibling sentinel under data/ and must be caught;
  - the Step 6 CLI gate compares the two baseline runs' logs as a control; a line counts as volatile
    only if it differs between them and looks like a time stamp or run id;
  - a gate drives the real Chapter 13 handoff (run_cycle -> _load_best_model_if_available ->
    post_draw_root_cause_analysis), with mutant handoff_bundle_dropped;
  - a wrong-checkpoint mutant: a second NN of the same family and width replaces the model inside
    the bundle, and A-NORM must fail;
  - the summary records the --baseline-rev actually used.

CPU only. Fixtures are synthetic: tests/test_ll_feature_settings.make_fixture plus the Step 3
worker (tests/ll_step3_runner.py), and four models trained by the real Step 5 CLI
(--trials 1, single-shot, storage inside --out). Reads no other worktree. Writes only under
--out, which must lie under <this tree>/data/.

  python3 -u -B tests/test_ll6_inference_bundle.py --out data/ll6_selftest_<stamp>
"""
import argparse
import hashlib
import importlib.util
import io
import json
import logging
import os
import re
import resource
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parent.parent
FAMILIES = ("neural_net", "xgboost", "lightgbm", "catboost")
BASELINE_REV = "7f4505aaccebdd284393eb3565e75d09d1c0f3b4"   # main before LL-6 (full sha)
A_NORM_ATOL = 1e-5            # declared before measurement (target units), as in ch14-learn W2
RESULTS = []


def gate(name, ok, detail=""):
    RESULTS.append({"gate": name, "ok": bool(ok), "detail": str(detail)[:600]})
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {str(detail)[:400]}", flush=True)
    return ok


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git_show(rev, rel):
    p = subprocess.run(["git", "-C", str(REPO), "show", f"{rev}:{rel}"], capture_output=True)
    return p.stdout if p.returncode == 0 else None


def load_as(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ENV_ROOTS = {"TMPDIR": "tmp", "XDG_CACHE_HOME": "cache", "MPLCONFIGDIR": "mpl", "CUDA_CACHE_PATH": "cuda_cache"}
USER_CACHE_DIRS = [Path.home() / ".cache" / "matplotlib", Path.home() / ".config" / "matplotlib"]


def redirect_env_roots(out):
    """Temp and cache roots under --out, set before any library import and inherited by every child
    (Beta X-LL6-BUNDLE R1 Q5: LightGBM's matplotlib import made /tmp/matplotlib-* when the default
    cache was not writable)."""
    import tempfile
    for k, d in ENV_ROOTS.items():
        (out / "_env" / d).mkdir(parents=True, exist_ok=True)
        os.environ[k] = str(out / "_env" / d)
    tempfile.tempdir = None
    return {k: os.environ[k] for k in ENV_ROOTS}


def boundary_snapshot(boundary):
    """Everything this suite must not write, outside the selected output subtree `boundary`:
    git status of this tree outside data/ (tracked, untracked, ignored), every path under data/
    with only `boundary` pruned (lstat size and mtime), new-file candidates /tmp/matplotlib-*, and
    the user's matplotlib cache and config dirs."""
    st = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--ignored", "--untracked-files=all",
                         "--", ".", ":(exclude)data"], capture_output=True, text=True).stdout.splitlines()
    walk = {}
    b = str(boundary)
    for root, dirs, files in os.walk(REPO / "data", followlinks=False):
        dirs[:] = [d for d in dirs if str(Path(root) / d) != b and not str(Path(root) / d).startswith(b + "/")]
        for n in files + [d for d in dirs if (Path(root) / d).is_symlink()]:
            q = Path(root) / n
            try:
                s_ = q.lstat()
                walk[str(q.relative_to(REPO))] = (s_.st_size, s_.st_mtime_ns)
            except FileNotFoundError:
                pass
    tmp = sorted(e.name for e in Path("/tmp").iterdir() if e.name.startswith("matplotlib-"))
    caches = {}
    for c in USER_CACHE_DIRS:
        if c.exists():
            for e in c.iterdir():
                s_ = e.lstat()
                caches[str(e)] = (s_.st_size, s_.st_mtime_ns)
    return {"git": sorted(st), "data": walk, "tmp_matplotlib": tmp, "user_caches": caches}


def boundary_diff(a, b):
    ch = []
    ch += [f"git: {x}" for x in sorted(set(b["git"]) ^ set(a["git"]))]
    for k in sorted(set(a["data"]) | set(b["data"])):
        if a["data"].get(k) != b["data"].get(k):
            ch.append(f"data: {k} {'new' if k not in a['data'] else ('gone' if k not in b['data'] else 'changed')}")
    ch += [f"/tmp: {x}" for x in sorted(set(b["tmp_matplotlib"]) - set(a["tmp_matplotlib"]))]
    for k in sorted(set(a["user_caches"]) | set(b["user_caches"])):
        if a["user_caches"].get(k) != b["user_caches"].get(k):
            ch.append(f"user cache: {k}")
    return ch


# ---------------------------------------------------------------------------
# fixtures (built once; mutant runs reuse them with --models-from)
# ---------------------------------------------------------------------------
def build_fixtures(out):
    sys.path.insert(0, str(REPO / "tests"))
    import test_ll_feature_settings as t1
    fx = out / "fx"
    t1.make_fixture(fx, seed=21, n=200)
    o = out / "step3_rows.json"
    subprocess.run([sys.executable, "-B", str(REPO / "tests/ll_step3_runner.py"), "--code-dir", str(REPO),
                    "--repo-root", str(REPO), "--fixture", str(fx), "--settings", "{}", "--out", str(o)],
                   check=True, capture_output=True)
    rows = [{"seed": x["seed"], "features": x["features"], "holdout_hits": x["holdout_hits"],
             "holdout_quality": x["holdout_quality"], "score": x["score"]}
            for x in json.load(open(o))["results"]]
    surv = out / "survivors.json"
    json.dump(rows, open(surv, "w"))
    hist = fx / "train.json"
    # a second fixture of the same schema for the alternate NN (wrong-checkpoint mutant)
    fx2 = out / "fx_alt"
    t1.make_fixture(fx2, seed=37, n=200)
    o2 = out / "step3_rows_alt.json"
    subprocess.run([sys.executable, "-B", str(REPO / "tests/ll_step3_runner.py"), "--code-dir", str(REPO),
                    "--repo-root", str(REPO), "--fixture", str(fx2), "--settings", "{}", "--out", str(o2)],
                   check=True, capture_output=True)
    surv_alt = out / "survivors_alt.json"
    json.dump([{"seed": x["seed"], "features": x["features"], "holdout_hits": x["holdout_hits"],
                "holdout_quality": x["holdout_quality"], "score": x["score"]}
               for x in json.load(open(o2))["results"]], open(surv_alt, "w"))

    def train(job):
        fam, tag, sv, hh = job
        R = out / f"models_{tag}"
        R.mkdir()
        shutil.copy(REPO / "train_single_trial.py", R / "train_single_trial.py")   # TODO-E2E-24
        extra = ["--normalize-features", "--use-leaky-relu"] if fam == "neural_net" else []
        env = dict(os.environ, PYTHONPATH=str(REPO), CUDA_VISIBLE_DEVICES="", S88_FORCE_LGBM_CPU="1",
                   PYTHONDONTWRITEBYTECODE="1")
        p = subprocess.run([sys.executable, "-u", "-B", str(REPO / "meta_prediction_optimizer_anti_overfit.py"),
                            "--survivors", str(sv), "--lottery-data", str(hh), "--trials", "1",
                            "--k-folds", "2", "--model-type", fam, "--device", "cpu",
                            "--storage", f"sqlite:///{R}/optuna.db", "--study-name", f"ll6_{tag}",
                            "--output-dir", str(R / "models/reinforcement")] + extra,
                           cwd=R, env=env, capture_output=True, text=True)
        (R / "run.log").write_text(p.stdout + p.stderr)
        if p.returncode != 0:
            raise RuntimeError(f"Step 5 {fam} failed rc={p.returncode}; see {R}/run.log")
        return R / "models/reinforcement"
    jobs = [(f, f, surv, hist) for f in FAMILIES] + [("neural_net", "neural_net_alt", surv_alt, fx2 / "train.json")]
    with ThreadPoolExecutor(len(jobs)) as ex:
        dirs = dict(zip([j[1] for j in jobs], ex.map(train, jobs)))
    return surv, hist, dirs


# ---------------------------------------------------------------------------
# Step 6 in-process (A-NORM) and through the CLI (default output)
# ---------------------------------------------------------------------------
def step6_in_process(pg, models_dir, rows):
    import numpy as np
    pg.GPU_AVAILABLE = False
    cfg = pg.PredictionConfig()
    cfg.models_dir = str(models_dir)
    log = logging.getLogger("ll6_step6")
    log.setLevel(logging.ERROR)
    gen = pg.PredictionGenerator(cfg, log)
    cap = {}
    orig = gen.model.predict

    def spy(X):
        y = orig(X)
        cap["y"] = np.asarray(y).copy()
        return y
    gen.model.predict = spy
    gen._build_prediction_pool(rows, [], 0)
    return np.asarray(cap["y"]).reshape(-1), gen


def step6_cli(out, tag, pg_file, code_dir, models_dir, surv, hist):
    d = out / "step6_cli" / tag
    d.mkdir(parents=True)
    cmd = [sys.executable, "-u", "-B", str(REPO / "tests/ll6_step6_runner.py"), "--pg-file", str(pg_file),
           "--repo-root", str(REPO), "--models-dir", str(models_dir), "--survivors", str(surv),
           "--history", str(hist), "--cwd", str(d / "cwd"), "--out", str(d / "result.json")]
    if code_dir:
        cmd += ["--code-dir", str(code_dir)]
    p = subprocess.run(cmd, capture_output=True, text=True,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"))
    (d / "runner.log").write_text(p.stdout + p.stderr)
    if p.returncode != 0:
        return {"runner_rc": p.returncode, "tail": (p.stdout + p.stderr)[-400:]}
    r = json.load(open(d / "result.json"))
    r["stdio"] = (p.stdout + p.stderr).replace(str(d), "<RUN>").splitlines()
    return r


# Declared stdio difference (not a behaviour change): torch's FutureWarning about
# torch.load(weights_only=False) names the file and line of the call, which LL-6 moved from
# prediction_generator.py into models/inference_bundle.py:81 (same call, same arguments).
TORCH_LOAD_WARNING = re.compile(r"^\S+\.py:\d+: (FutureWarning: You are using `torch\.load`)")


def norm_stdio(lines):
    return [TORCH_LOAD_WARNING.sub(r"<torch.load call site>: \1", l) for l in lines]


def flatten(o, pre=""):
    if isinstance(o, dict):
        r = {}
        for k, v in o.items():
            r.update(flatten(v, f"{pre}/{k}"))
        return r
    if isinstance(o, list) and o and isinstance(o[0], (dict, list)):
        r = {}
        for i, v in enumerate(o):
            r.update(flatten(v, f"{pre}[{i}]"))
        return r
    return {pre: o}


def independent_forward(b, Z):
    """The attribution path's own forward pass: the object the bundle hands to the attribution
    backends (bundle.attribution_model), on the bundle's model input, in target units."""
    import numpy as np
    fam = b.model_type
    m = b.attribution_model
    if fam == "neural_net":
        import torch
        # The bundle hands over the module as loaded, in TRAINING mode (dropout, BatchNorm batch
        # statistics). Every consumer puts it in eval mode first: Step 6's wrapper.predict
        # (neural_net_wrapper.py:350) and Chapter 13's NN backend (per_survivor_attribution.py:91,
        # restored at :121). So does this forward, and it restores the mode afterwards.
        was = m.training
        m.eval()
        try:
            with torch.no_grad():
                f = m(torch.from_numpy(np.ascontiguousarray(Z, dtype=np.float32))).cpu().numpy().reshape(-1)
        finally:
            if was:
                m.train()
        f = f.astype(np.float64)
        if b.y_mean is not None:
            f = f * float(b.y_std) + float(b.y_mean)
        return f
    if fam == "xgboost":
        import xgboost as xgb
        return np.asarray(m.predict(xgb.DMatrix(Z))).reshape(-1)
    return np.asarray(m.predict(Z)).reshape(-1)          # lightgbm Booster, CatBoostRegressor


# ---------------------------------------------------------------------------
# mutants: one exact replacement in a copy, code dir first on sys.path
# ---------------------------------------------------------------------------
MARK = '__import__("pathlib").Path(__import__("os").environ["LL6_MUTANT_MARK"]).touch()'
# (id, file, old, new, count, insert_marker, target gates)
MUTANTS = [
    ("scaler_log_changed", "models/inference_bundle.py",
     'logger.info(f"[CAT-B] NN scaler loaded: {len(scaler_mean)} features")',
     'logger.info(f"[CAT-B] NN scaler loaded: {len(scaler_mean)} feature(s)")', 1, True,
     ["LL6.step6_scaler_identical_to_base", "LL6.step6_cli_default_output_identical_to_base"]),
    ("scaler_flag_dropped", "models/inference_bundle.py",
     "out.update(scaler_mean=scaler_mean, scaler_scale=scaler_scale, normalize_features=True)",
     "out.update(scaler_mean=scaler_mean, scaler_scale=scaler_scale, normalize_features=False)", 1, True,
     ["LL6.step6_scaler_identical_to_base", "LL6.step6_cli_default_output_identical_to_base"]),
    ("clamp_threshold_changed", "models/inference_bundle.py",
     "zero_mask = scaler_scale < 1e-12", "zero_mask = scaler_scale < 1e9", 1, True,
     ["LL6.step6_scaler_identical_to_base", "LL6.step6_cli_default_output_identical_to_base"]),
    ("bundle_transform_skips_scaler", "models/inference_bundle.py",
     "X = (X - self.scaler_mean) / self.scaler_scale", "X = X", 1, True,
     ["LL6.A-NORM_all_four_families"]),
    ("bundle_names_reversed", "models/inference_bundle.py",
     'return list(fs.get("per_seed_feature_names", fs.get("feature_names", [])) or [])',
     'return list(reversed(list(fs.get("per_seed_feature_names", fs.get("feature_names", [])) or [])))',
     1, True, ["LL6.A-NORM_all_four_families", "LL6.c13_canonical_loader_all_four"]),
    ("scaler_source_default_as_written", "models/inference_bundle.py",
     'scaler_source: str = "resolved",', 'scaler_source: str = "as_written",', 1, False,
     ["LL6.relocated_dir_bundle_keeps_scaler"]),
    ("step6_flag_not_set", "prediction_generator.py",
     "self._normalize_features = True", "self._normalize_features = False", 1, True,
     ["LL6.step6_scaler_identical_to_base", "LL6.step6_cli_default_output_identical_to_base"]),
    ("c13_default_canonical", "chapter_13_orchestrator.py",
     "c13_model_loader: str = 'legacy',", "c13_model_loader: str = 'canonical',", 1, False,
     ["LL6.c13_default_is_legacy", "LL6.c13_default_construction_loads_as_base"]),
    ("c13_input_unscaled", "chapter_13_orchestrator.py",
     "return bundle.transform(x.reshape(1, -1))[0]", "return x.reshape(-1)", 1, True,
     ["LL6.c13_root_cause_canonical_uses_scaled_named_input"]),
    ("c13_canonical_names_dropped", "chapter_13_orchestrator.py",
     '"feature_names": list(bundle.feature_names), "bundle": bundle,',
     '"feature_names": None, "bundle": bundle,', 1, False,
     ["LL6.c13_canonical_loader_all_four"]),
    ("c13_legacy_input_rerouted", "chapter_13_orchestrator.py",
     "if bundle is not None", "if True", 2, False,
     ["LL6.c13_root_cause_legacy_identical_to_base"]),
    ("handoff_bundle_dropped", "chapter_13_orchestrator.py",        # Beta R1 Q5: the run_cycle handoff
     "bundle=model_info.get('bundle'),  # [LL-6] None for legacy", "bundle=None,  # [LL-6] None for legacy",
     1, False, ["LL6.c13_run_cycle_hands_bundle_to_root_cause"]),
    ("wrong_checkpoint_same_family", "models/inference_bundle.py",  # Beta R1 Q5, carried from W2
     "model, meta2 = load_model_from_sidecar(models_dir=models_dir, device=device)",
     'model, meta2 = load_model_from_sidecar(models_dir=(__import__("os").environ["LL6_ALT_NN_DIR"] '
     'if "models_neural_net/" in str(models_dir) + "/" else models_dir), device=device)',
     1, True, ["LL6.A-NORM_all_four_families"]),
]
# extra environment for a mutant run, from the fixture source
MUTANT_ENV = {"wrong_checkpoint_same_family":
              lambda src: {"LL6_ALT_NN_DIR": str(src / "models_neural_net_alt" / "models" / "reinforcement")}}


def make_mutant_dir(root, rel, old, new, count, marker):
    src = (REPO / rel).read_text()
    n = src.count(old)
    if n != count:
        return None, f"anchor found {n} times, want {count}"
    if marker:
        out_lines, done = [], 0
        for line in src.split("\n"):
            if old in line and done < count:
                indent = line[:len(line) - len(line.lstrip())]
                out_lines.append(indent + MARK)
                done += 1
            out_lines.append(line)
        src2 = "\n".join(out_lines).replace(old, new)
    else:
        src2 = src.replace(old, new)
    code = root / "code"
    code.mkdir(parents=True)
    parts = Path(rel).parts
    if len(parts) == 2:                       # models/<file>: a models/ dir of symlinks plus the copy
        (code / parts[0]).mkdir()
        for e in (REPO / parts[0]).iterdir():
            if e.name != parts[1] and e.name != "__pycache__":
                (code / parts[0] / e.name).symlink_to(e)
    (code / rel).write_text(src2)
    return code, f"sha {sha(REPO / rel)[:12]} -> {sha(code / rel)[:12]}"


def run_mutants(out, jobs, models_src, green, baseline_rev, boundary):
    res = []

    def one(m):
        mid, rel, old, new, count, marker, targets = m
        root = out / "mutants" / mid
        unmut = {t: green.get(t) for t in targets}
        if not all(v is True for v in unmut.values()):
            return {"id": mid, "killed": False, "targets": targets, "unmutated": unmut,
                    "detail": "a target gate is not green in the unmutated run: no credit"}
        code, how = make_mutant_dir(root, rel, old, new, count, marker)
        if code is None:
            return {"id": mid, "killed": False, "detail": how}
        mark = root / "mark"
        extra = MUTANT_ENV[mid](models_src) if mid in MUTANT_ENV else {}
        p = subprocess.run([sys.executable, "-u", "-B", str(Path(__file__).resolve()), "--out", str(root / "run"),
                            "--code-dir", str(code), "--models-from", str(models_src), "--no-mutants",
                            "--only", ",".join(targets), "--baseline-rev", baseline_rev,
                            "--boundary", str(boundary)],
                           capture_output=True, text=True,
                           env=dict(os.environ, LL6_MUTANT_MARK=str(mark), CUDA_VISIBLE_DEVICES="",
                                    PYTHONDONTWRITEBYTECODE="1", **extra))
        (root / "run.log").write_text(p.stdout + p.stderr)
        try:
            g = {r["gate"]: r["ok"] for r in json.load(open(root / "run" / "selftest_summary.json"))["gates"]}
        except Exception as e:
            return {"id": mid, "killed": False, "detail": f"no summary ({e}); rc {p.returncode}"}
        red = [t for t in targets if g.get(t) is False]
        killed = len(red) == len(targets) and (mark.exists() or not marker)
        return {"id": mid, "file": rel, "killed": killed, "targets_red": red, "targets": targets,
                "unmutated": unmut, "env": sorted(extra), "marker": ("hit" if mark.exists() else "absent") if marker else "n/a (no statement position)",
                "applied": how}
    with ThreadPoolExecutor(jobs) as ex:
        for r in ex.map(one, MUTANTS):
            res.append(r)
            print(f"[{'PASS' if r['killed'] else 'FAIL'}] MUTANT LL6.{r['id']} killed={r['killed']} "
                  f"unmutated={r.get('unmutated')} red={r.get('targets_red')} marker={r.get('marker')} "
                  f"{r.get('applied', r.get('detail'))}",
                  flush=True)
    return res


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline-rev", default=BASELINE_REV,
                    help="full sha of main before LL-6 (default 7f4505a...); HEAD and short shas are refused")
    ap.add_argument("--code-dir", default=None, help="mutant runs only: put this dir first on sys.path")
    ap.add_argument("--models-from", default=None, help="reuse fixtures and models from an earlier --out")
    ap.add_argument("--only", default=None, help="comma list of gates to run (mutant runs)")
    ap.add_argument("--no-mutants", action="store_true")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--boundary", default=None,
                    help="the output subtree the no-write gate excludes (default: --out). It must contain "
                         "--out; use it for a run directory that holds --out and the run's own stdout log. "
                         "Mutant runs get the parent run's boundary.")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if not str(out).startswith(str(REPO / "data") + "/"):
        raise SystemExit("--out must lie under <this tree>/data/")
    if not re.fullmatch(r"[0-9a-f]{40}", a.baseline_rev):
        raise SystemExit("--baseline-rev must be a full 40-hex sha (never HEAD, a branch or a short sha)")
    out.mkdir(parents=True)
    RUN_INFO["baseline_rev"] = a.baseline_rev
    RUN_INFO["env_roots"] = redirect_env_roots(out)    # before numpy, torch, lightgbm or any child
    import numpy as np
    boundary = Path(a.boundary).resolve() if a.boundary else out
    if out != boundary and not str(out).startswith(str(boundary) + "/"):
        raise SystemExit("--out must lie inside --boundary")
    if not str(boundary).startswith(str(REPO / "data") + "/"):
        raise SystemExit("--boundary must lie under <this tree>/data/")
    RUN_INFO["boundary"] = str(boundary)
    only = set(a.only.split(",")) if a.only else None
    want = (lambda g: only is None or g in only)
    code_dir = Path(a.code_dir).resolve() if a.code_dir else None
    sys.path.insert(0, str(REPO))
    if code_dir:
        sys.path.insert(0, str(code_dir))
    t0 = time.time()
    guard_before = boundary_snapshot(boundary)
    status_before = boundary

    # ---- baseline precondition -------------------------------------------
    base = out / "base"
    base.mkdir()
    pg_b, c13_b = git_show(a.baseline_rev, "prediction_generator.py"), git_show(a.baseline_rev, "chapter_13_orchestrator.py")
    ib_b = git_show(a.baseline_rev, "models/inference_bundle.py")
    under = lambda rel: (code_dir / rel if code_dir and (code_dir / rel).exists() else REPO / rel)
    pg_now, c13_now = under("prediction_generator.py").read_text(), under("chapter_13_orchestrator.py").read_text()
    pre_ok = (pg_b is not None and c13_b is not None and ib_b is None
              and b"models.inference_bundle" not in pg_b and b"c13_model_loader" not in c13_b
              and "models.inference_bundle" in pg_now and "c13_model_loader" in c13_now)
    gate("LL6.baseline_precondition", pre_ok,
         f"baseline {a.baseline_rev[:12]} must lack LL-6 (no inference_bundle, no c13_model_loader) and the "
         f"tree under test must have it; otherwise every *_identical_to_base gate compares the new code with itself")
    if not pre_ok:
        return finish(out, t0, [], guard_before, status_before)
    (base / "pg_base.py").write_bytes(pg_b)           # renamed: never importable as the real module names
    (base / "c13_base.py").write_bytes(c13_b)

    # ---- fixtures ---------------------------------------------------------
    if a.models_from:
        src = Path(a.models_from).resolve()
        surv, hist = src / "survivors.json", src / "fx" / "train.json"
        dirs = {f: src / f"models_{f}" / "models/reinforcement" for f in FAMILIES + ("neural_net_alt",)}
    else:
        surv, hist, dirs = build_fixtures(out)
    t_fix = time.time() - t0
    rows = json.load(open(surv))
    feats = [r["features"] for r in rows]

    # Import order matters: prediction_generator puts its own directory first on sys.path when
    # imported (prediction_generator.py:57), so the modules a mutant code dir replaces are
    # imported before it and are then the ones every later import (and Step 6) uses.
    import models.inference_bundle as ib
    import chapter_13_orchestrator as c13
    import prediction_generator as pg
    for _m, _rel in ((ib, "models/inference_bundle.py"), (c13, "chapter_13_orchestrator.py"),
                     (pg, "prediction_generator.py")):
        if Path(_m.__file__).resolve() != under(_rel).resolve() and Path(_m.__file__) != under(_rel):
            raise SystemExit(f"import resolved {_rel} to {_m.__file__}, not {under(_rel)}")
    pg_base = load_as("ll6_pg_base", base / "pg_base.py")
    c13_base = load_as("ll6_c13_base", base / "c13_base.py")

    # ---- G1 Step 6 scaler: same attributes and log lines as the baseline ----
    nn_dir = dirs["neural_net"]
    ck = json.load(open(nn_dir / "best_model.meta.json"))["checkpoint_path"]
    if want("LL6.step6_scaler_identical_to_base"):
        def scaler_run(mod):
            class S:
                pass
            s = S()
            s._scaler_mean = s._scaler_scale = None
            s._normalize_features = False
            buf = io.StringIO()
            lg = logging.getLogger(f"ll6_scaler_{id(mod)}")
            lg.handlers = [logging.StreamHandler(buf)]
            lg.setLevel(logging.INFO)
            lg.propagate = False
            s.logger = lg
            mod.PredictionGenerator._load_nn_scaler(s, ck)
            return s, buf.getvalue()
        s_new, log_new = scaler_run(pg)
        s_old, log_old = scaler_run(pg_base)
        gate("LL6.step6_scaler_identical_to_base",
             s_old._normalize_features is True and s_new._normalize_features == s_old._normalize_features
             and np.array_equal(s_new._scaler_mean, s_old._scaler_mean)
             and np.array_equal(s_new._scaler_scale, s_old._scaler_scale) and log_new == log_old,
             f"normalize {s_new._normalize_features}/{s_old._normalize_features}; mean/scale arrays equal "
             f"{np.array_equal(s_new._scaler_mean, s_old._scaler_mean)}/"
             f"{np.array_equal(s_new._scaler_scale, s_old._scaler_scale)}; "
             f"{len(log_new.splitlines())} vs {len(log_old.splitlines())} log lines, equal {log_new == log_old}")

    # ---- G2 Step 6 CLI default output, 4 families, against the baseline ----
    if want("LL6.step6_cli_default_output_identical_to_base"):
        pg_file = under("prediction_generator.py")
        detail, ok = {}, True
        for fam in FAMILIES:
            new = step6_cli(out, f"{fam}_new", pg_file, code_dir, dirs[fam], surv, hist)
            b1 = step6_cli(out, f"{fam}_base1", base / "pg_base.py", None, dirs[fam], surv, hist)
            b2 = step6_cli(out, f"{fam}_base2", base / "pg_base.py", None, dirs[fam], surv, hist)
            if any("runner_rc" in x for x in (new, b1, b2)) or b1["result"] is None:
                ok = False
                detail[fam] = {"error": [x.get("tail") for x in (new, b1, b2) if "runner_rc" in x]}
                continue
            f1, f2, fn = flatten(b1["result"]), flatten(b2["result"]), flatten(new["result"] or {})
            volatile = sorted(k for k in set(f1) | set(f2) if f1.get(k) != f2.get(k))
            diff = sorted(k for k in set(f1) | set(fn) if k not in volatile and f1.get(k) != fn.get(k))
            # log control (Beta R1 Q5): a line is volatile only if it differs between the two baseline
            # runs AND looks like a clock or run id; anything else must be equal on all three runs
            from collections import Counter
            c1, c2 = Counter(b1["log_lines"]), Counter(b2["log_lines"])
            vol_lines = set((c1 - c2) + (c2 - c1))
            vol_bad = sorted(l for l in vol_lines if not VOLATILE_LINE.search(l))
            keep = lambda r: [l for l in r["log_lines"] if l not in vol_lines]
            logs_eq = keep(new) == keep(b1) and keep(b1) == keep(b2)
            # non-vacuity of the log check: it must compare lines (it compared none before R1's fix)
            n_log = len(keep(b1))
            # stdout/stderr too, with only the declared torch.load call-site difference normalised
            sn, s1, s2 = norm_stdio(new["stdio"]), norm_stdio(b1["stdio"]), norm_stdio(b2["stdio"])
            stdio_eq = sn == s1 and s1 == s2
            n_pred = len((b1["result"] or {}).get("predictions", []))
            fam_ok = (new["rc"] == b1["rc"] == b2["rc"] and not diff and logs_eq and n_log > 0 and stdio_eq
                      and not vol_bad and n_pred > 0
                      and all(re.search(r"timestamp|run_id|created|_at$|generated", k) for k in volatile))
            ok &= fam_ok
            detail[fam] = {"ok": fam_ok, "keys_compared": len(set(f1) - set(volatile)), "differ": diff[:6],
                           "volatile_in_control": volatile, "log_lines_compared": len(keep(b1)),
                           "volatile_log_lines": len(vol_lines), "volatile_not_clock_like": vol_bad[:3],
                           "log_lines_equal": logs_eq, "stdio_lines": len(s1), "stdio_equal": stdio_eq,
                           "stdio_first_diff": next(([x, y] for x, y in zip(sn, s1) if x != y), None),
                           "predictions": n_pred}
        gate("LL6.step6_cli_default_output_identical_to_base", ok, detail)

    # ---- G3 A-NORM, four families -----------------------------------------
    bundles = {}
    if want("LL6.A-NORM_all_four_families") or want("LL6.c13_canonical_loader_all_four") \
            or want("LL6.c13_root_cause_canonical_uses_scaled_named_input"):
        for fam in FAMILIES:
            bundles[fam] = ib.load_inference_bundle(str(dirs[fam]))
    if want("LL6.A-NORM_all_four_families"):
        an = {}
        for fam in FAMILIES:
            y6, _ = step6_in_process(pg, dirs[fam], rows)
            b = bundles[fam]
            Z = b.transform(b.matrix(feats))
            an[fam] = {"step6_vs_attribution_forward": float(np.max(np.abs(y6 - independent_forward(b, Z)))),
                       "step6_vs_bundle_predict": float(np.max(np.abs(y6 - b.predict(b.matrix(feats))))),
                       "normalize": b.normalize_features}
        gate("LL6.A-NORM_all_four_families",
             all(v["step6_vs_attribution_forward"] <= A_NORM_ATOL and v["step6_vs_bundle_predict"] <= A_NORM_ATOL
                 for v in an.values()) and an["neural_net"]["normalize"],
             {f: f"{v['step6_vs_attribution_forward']:.2e}" for f, v in an.items()} | {"atol": A_NORM_ATOL})

    # ---- G4 relocated NN directory. [1b, 2026-11-26] Step 6 now reads the scaler from the
    # file the weights were loaded from (Beta X-1B-STEP6-NN R1 Q2), so Step 6 on a relocated
    # dir equals the default (resolved) bundle and differs from the as_written replay.
    if want("LL6.relocated_dir_bundle_keeps_scaler"):
        reloc = out / "relocated_nn"
        shutil.copytree(nn_dir, reloc)
        m = json.load(open(reloc / "best_model.meta.json"))
        m["checkpoint_path"] = "/nonexistent_moved_dir/best_model.pth"
        (reloc / "best_model.meta.json").write_text(json.dumps(m))
        y6r, g6r = step6_in_process(pg, reloc, rows)
        br = ib.load_inference_bundle(str(reloc))                       # default scaler_source
        bw = ib.load_inference_bundle(str(reloc), scaler_source="as_written")
        y_res, y_asw = br.predict(br.matrix(feats)), bw.predict(bw.matrix(feats))
        gap = float(np.max(np.abs(y6r - y_res)))
        gap_asw = float(np.max(np.abs(y6r - y_asw)))
        gate("LL6.relocated_dir_bundle_keeps_scaler",
             g6r._normalize_features is True and gap <= A_NORM_ATOL
             and gap_asw > 1e3 * A_NORM_ATOL and br.normalize_features and br.scaler_source == "resolved"
             and not bw.normalize_features,
             f"Step 6 normalize={g6r._normalize_features} (scaler read from the resolved file); default bundle "
             f"scaler_source={br.scaler_source} normalize={br.normalize_features}; Step 6 vs bundle(default) "
             f"max |diff| {gap:.2e}; vs bundle(as_written, unscaled) {gap_asw:.3g}")

    # ---- Chapter 13 -------------------------------------------------------
    def c13_load(mod, models_dir, loader=None, construct=False):
        if construct:
            o = mod.Chapter13Orchestrator.__new__(mod.Chapter13Orchestrator)
            init = mod.Chapter13Orchestrator.__init__
            import inspect
            dflt = inspect.signature(init).parameters.get("c13_model_loader")
            if dflt is not None:                     # the default as a caller that passes nothing gets it
                o.c13_model_loader = dflt.default
                o.c13_models_dir = str(models_dir)
        else:
            o = object.__new__(mod.Chapter13Orchestrator)
            if loader:
                o.c13_model_loader = loader
                o.c13_models_dir = str(models_dir)
        cwd = os.getcwd()
        os.chdir(Path(models_dir).parent.parent)      # legacy reads models/reinforcement under cwd
        try:
            r = o._load_best_model_if_available()
        finally:
            os.chdir(cwd)
        if r is None:
            return None
        return {"model_type": r["model_type"], "feature_names": r.get("feature_names"),
                "model_class": type(r["model"]).__name__, "bundle": "bundle" in r}

    legacy_old = None
    if want("LL6.c13_legacy_loader_identical_to_base") or want("LL6.c13_default_construction_loads_as_base") \
            or want("LL6.c13_canonical_loader_all_four"):
        legacy_old = {f: c13_load(c13_base, dirs[f]) for f in FAMILIES}
    if want("LL6.c13_legacy_loader_identical_to_base"):
        legacy_new = {f: c13_load(c13, dirs[f]) for f in FAMILIES}
        gate("LL6.c13_legacy_loader_identical_to_base", legacy_new == legacy_old,
             {f: (None if v is None else (v["model_class"], v["feature_names"] is None)) for f, v in legacy_new.items()})
    if want("LL6.c13_default_construction_loads_as_base"):
        dflt_new = {f: c13_load(c13, dirs[f], construct=True) for f in FAMILIES}
        gate("LL6.c13_default_construction_loads_as_base", dflt_new == legacy_old,
             "the loader a caller gets by passing nothing loads exactly what the baseline loads")
    if want("LL6.c13_default_is_legacy"):
        import inspect
        d = inspect.signature(c13.Chapter13Orchestrator.__init__).parameters["c13_model_loader"].default
        cli = re.search(r'add_argument\("--c13-model-loader".*?default="(\w+)"', c13_now, re.S)
        gate("LL6.c13_default_is_legacy", d == "legacy" and cli is not None and cli.group(1) == "legacy"
             and tuple(c13.C13_MODEL_LOADERS) == ("legacy", "canonical"),
             f"constructor default {d!r}; CLI default {cli and cli.group(1)!r}; choices {c13.C13_MODEL_LOADERS}")
    if want("LL6.c13_canonical_loader_all_four"):
        canon = {f: c13_load(c13, dirs[f], "canonical") for f in FAMILIES}
        side = {f: json.load(open(dirs[f] / "best_model.meta.json")).get("feature_schema", {}) for f in FAMILIES}
        side_names = {f: list(s.get("per_seed_feature_names", s.get("feature_names", []))) for f, s in side.items()}
        differs = [f for f in FAMILIES if canon[f] != legacy_old[f]]
        gate("LL6.c13_canonical_loader_all_four",
             all(v and v["bundle"] and v["feature_names"] == side_names[f] for f, v in canon.items())
             and "neural_net" in differs,
             f"canonical: { {f: (v and v['model_class']) for f, v in canon.items()} }; non-vacuity: canonical differs "
             f"from the baseline's loader for {differs} (legacy NN = {legacy_old['neural_net']})")

    def root_cause(mod, preds, model, fam, names, bundle, tag):
        d = out / "c13_cwd" / tag                     # the archive goes under --out
        d.mkdir(parents=True)
        cwd = os.getcwd()
        os.chdir(d)
        try:
            o = object.__new__(mod.Chapter13Orchestrator)
            kw = {"bundle": bundle} if bundle is not None else {}
            return mod.Chapter13Orchestrator.post_draw_root_cause_analysis(
                o, {"draw_id": "ll6_fixture"}, preds, model, fam, names, **kw)
        finally:
            os.chdir(cwd)

    if want("LL6.c13_root_cause_legacy_identical_to_base"):
        fam = "lightgbm"
        cwd = os.getcwd()
        os.chdir(dirs[fam].parent.parent)
        try:
            lm = object.__new__(c13_base.Chapter13Orchestrator)._load_best_model_if_available()
        finally:
            os.chdir(cwd)
        names = json.load(open(dirs[fam] / "best_model.meta.json"))["feature_schema"]["feature_names"]
        preds = [{"seed": r["seed"], "rank": i + 1, "hit": i in (2, 7),
                  "features": [float(r["features"].get(n, 0.0)) for n in names]} for i, r in enumerate(rows[:20])]
        r_new = root_cause(c13, preds, lm["model"], fam, names, None, "legacy_new")
        r_old = root_cause(c13_base, preds, lm["model"], fam, names, None, "legacy_base")
        strip = lambda r: {k: v for k, v in (r or {}).items() if k != "timestamp"}
        gate("LL6.c13_root_cause_legacy_identical_to_base",
             r_old is not None and strip(r_new) == strip(r_old) and r_old["attribution_success"]["missed"] == 18,
             f"bundle=None (legacy): result equal to the baseline apart from timestamp: {strip(r_new) == strip(r_old)}; "
             f"diagnosis {r_old and r_old['diagnosis']}; archives under --out: "
             f"{len(list((out / 'c13_cwd').rglob('root_cause_*.json')))}")

    if want("LL6.c13_root_cause_canonical_uses_scaled_named_input"):
        bnn = bundles["neural_net"]
        preds = [{"seed": r["seed"], "rank": i + 1, "hit": i in (2, 7), "features": dict(r["features"])}
                 for i, r in enumerate(rows[:20])]
        captured = []
        orig_psa = c13.per_survivor_attribution

        def spy_psa(model, model_type, features, feature_names):
            captured.append(np.asarray(features))
            return orig_psa(model, model_type, features, feature_names)
        c13.per_survivor_attribution = spy_psa
        try:
            res = root_cause(c13, preds, bnn.attribution_model, "neural_net", bnn.feature_names, bnn, "canonical_nn")
        finally:
            c13.per_survivor_attribution = orig_psa
        want_z = bnn.transform(bnn.matrix([p["features"] for p in preds]))
        gs = sorted(captured, key=lambda v: float(v.sum()))
        ws = sorted(list(want_z), key=lambda v: float(v.sum()))
        gate("LL6.c13_root_cause_canonical_uses_scaled_named_input",
             res is not None and len(captured) == 20 and all(np.array_equal(g, w) for g, w in zip(gs, ws))
             and res["attribution_success"] == {"missed": 18, "hit": 2},
             f"{len(captured)} attributions on bundle.transform(bundle.vector(features)); diagnosis "
             f"{res and res['diagnosis']} (observe-only, rule unchanged)")

    # ---- the real Chapter 13 handoff (Beta R1 Q5): run_cycle -> loader -> root cause --------
    if want("LL6.c13_run_cycle_hands_bundle_to_root_cause"):
        import types
        bnn2 = ib.load_inference_bundle(str(dirs["neural_net"]))
        preds = [{"seed": r["seed"], "rank": i + 1, "hit": i in (2, 7), "features": dict(r["features"])}
                 for i, r in enumerate(rows[:20])]
        cap = {"inputs": []}

        class _StopCycle(BaseException):       # leaves run_cycle before triggers, LLM or cycle log
            pass
        o = object.__new__(c13.Chapter13Orchestrator)
        o.c13_model_loader, o.c13_models_dir = "canonical", str(dirs["neural_net"])
        o._detect_hit_regression = lambda d: True
        o.load_predictions_from_disk = lambda expected_draw_id=None: preds

        def spy_rc(**kw):
            cap["kw"] = kw
            cap["result"] = c13.Chapter13Orchestrator.post_draw_root_cause_analysis(o, **kw)
            raise _StopCycle()
        o.post_draw_root_cause_analysis = spy_rc
        orig = (c13.generate_diagnostics, c13.save_diagnostics, c13.per_survivor_attribution,
                sys.modules.get("database_system"))
        fake_db = types.ModuleType("database_system")

        class _NoDB:
            def __init__(self, *a_, **k_):
                raise RuntimeError("ll6 test: no database")
        fake_db.DistributedPRNGDatabase = _NoDB
        d = out / "c13_cwd" / "run_cycle"
        d.mkdir(parents=True)
        cwd = os.getcwd()
        stopped = False
        try:
            c13.generate_diagnostics = lambda *a_, **k_: {"draw_id": "ll6_handoff", "summary_flags": []}
            c13.save_diagnostics = lambda *a_, **k_: None
            c13.per_survivor_attribution = lambda model, model_type, features, feature_names: (
                cap["inputs"].append(np.asarray(features)) or orig[2](model, model_type, features, feature_names))
            sys.modules["database_system"] = fake_db
            os.chdir(d)
            try:
                o.run_cycle()
            except _StopCycle:
                stopped = True
        finally:
            os.chdir(cwd)
            c13.generate_diagnostics, c13.save_diagnostics, c13.per_survivor_attribution = orig[:3]
            if orig[3] is None:
                sys.modules.pop("database_system", None)
            else:
                sys.modules["database_system"] = orig[3]
        kw = cap.get("kw", {})
        hb = kw.get("bundle")
        want_z = bnn2.transform(bnn2.matrix([p["features"] for p in preds]))
        gs = sorted(cap["inputs"], key=lambda v: float(np.asarray(v, dtype=np.float64).sum()))
        ws = sorted(list(want_z), key=lambda v: float(v.sum()))
        ok_in = len(gs) == 20 and all(np.asarray(g).shape == w.shape and np.array_equal(g, w) for g, w in zip(gs, ws))
        res = cap.get("result")
        gate("LL6.c13_run_cycle_hands_bundle_to_root_cause",
             stopped and hb is not None and getattr(hb, "checkpoint_hash", None) == bnn2.checkpoint_hash
             and kw.get("feature_names") == list(bnn2.feature_names) and ok_in and res is not None
             and res["attribution_success"] == {"missed": 18, "hit": 2},
             f"run_cycle reached root cause: {stopped}; bundle handed over: {hb is not None} "
             f"(checkpoint {getattr(hb, 'checkpoint_hash', '')[:12]}); {len(gs)} attribution inputs equal "
             f"bundle.transform(bundle.vector(features)): {ok_in}; result {res and res['attribution_success']}")

    mutants = []
    if not a.no_mutants and code_dir is None:
        green = {r["gate"]: r["ok"] for r in RESULTS}
        mutants = run_mutants(out, a.jobs, Path(a.models_from).resolve() if a.models_from else out, green,
                              a.baseline_rev, boundary)
    # negative control for the no-write gate (Beta R1 Q5): a sibling sentinel under data/, outside --out
    if code_dir is None and want("LL6.no_write_gate_catches_sibling_write"):
        s0 = boundary_snapshot(boundary)
        sent = REPO / "data" / f"ll6_nowrite_control_{os.getpid()}_{time.time_ns()}"
        sent.mkdir()
        (sent / "sentinel.txt").write_text("ll6 no-write negative control\n")
        s1 = boundary_snapshot(boundary)
        shutil.rmtree(sent)
        s2 = boundary_snapshot(boundary)
        fake = dict(s2, tmp_matplotlib=s2["tmp_matplotlib"] + ["matplotlib-ll6probe"])
        caught, cleared, tmp_caught = boundary_diff(s0, s1), boundary_diff(s0, s2), boundary_diff(s2, fake)
        gate("LL6.no_write_gate_catches_sibling_write",
             any(str(sent.relative_to(REPO)) in c for c in caught) and not cleared and tmp_caught,
             f"sibling sentinel {sent.relative_to(REPO)} seen: {caught[:2]}; after removal clean: {not cleared}; "
             f"a new /tmp/matplotlib-* entry (synthetic listing) seen: {tmp_caught}")
    return finish(out, t0, mutants, guard_before, status_before, t_fix)


VOLATILE_LINE = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}|\d{8}_\d{6}|timestamp|run_id")
RUN_INFO = {}


def finish(out, t0, mutants, guard_before, status_before, t_fix=None):
    boundary = status_before
    guard_after = boundary_snapshot(boundary)
    ch = boundary_diff(guard_before, guard_after)
    gate("LL6.no_write_outside_out", not ch,
         f"outside {Path(boundary).relative_to(REPO)}: git status ({len(guard_after['git'])} lines), "
         f"{len(guard_after['data'])} paths under data/, /tmp/matplotlib-*, user matplotlib dirs unchanged; "
         f"changes: {ch[:5]}")
    n_ok = sum(r["ok"] for r in RESULTS)
    k_ok = sum(m["killed"] for m in mutants)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    rss_c = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
    summ = {"gates": RESULTS, "mutants": mutants, "gates_pass": n_ok, "gates_total": len(RESULTS),
            "mutants_killed": k_ok, "mutants_total": len(mutants), "wall_s": round(time.time() - t0, 1),
            "fixture_s": None if t_fix is None else round(t_fix, 1),
            "rss_mb": {"self": round(rss, 1), "largest_child": round(rss_c, 1)},
            "baseline_rev": RUN_INFO.get("baseline_rev"), "baseline_rev_default": BASELINE_REV,
            "boundary": RUN_INFO.get("boundary"), "env_roots": RUN_INFO.get("env_roots"), "repo": str(REPO)}
    json.dump(summ, open(out / "selftest_summary.json", "w"), indent=1, default=str)
    print(f"{n_ok}/{len(RESULTS)} gates pass; mutants killed {k_ok}/{len(mutants)}; wall {summ['wall_s']} s; "
          f"RSS self {rss:.0f} MB, largest child {rss_c:.0f} MB", flush=True)
    return 0 if n_ok == len(RESULTS) and k_ok == len(mutants) else 1


if __name__ == "__main__":
    sys.exit(main())
