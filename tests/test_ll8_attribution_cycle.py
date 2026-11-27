#!/usr/bin/env python3
"""
tests/test_ll8_attribution_cycle.py - self-test for LL-8, scripts/diagnostics/attribution_cycle.py
(plan PLAN_LAND_LL7_LL8_20261126 §5; branch ch14-learn blob f4a8adbd plus the landing changes).

Gates: the branch's end-to-end checks (A-NORM inside, a planted per-seed signal ranks first, per-run
columns are context only, a sparse cohort abstains, bands are disjoint, advisory label and LL-3
variants) and the landing changes: required checkpoint pin; wrong pin on a same-family, same-width NN
refused; G-Q4-SCALER (a relocated NN with its written checkpoint path missing is refused unless
scaler_source "as_written", which runs and says so in the report); miss-report default is this
tree's copy; model family and flag id checks; the function entry point and the CLI's exit status and
result file; no write outside --out-dir; nothing in the tree imports the script.

R2 fix pass (Beta X-LL7-LL8 R1): Q3 Step 6 evidence (Beta's fixture: Step 6 scores first, the model is
relocated second, the wrong historical transform is refused despite a matching pin and a green internal
A-NORM; as_written without evidence refused; a scaler-state mismatch refused; a resolved run that matches
saved scores runs); Q5 result file distinct from every output, never overwritten, publication and path
failures returned as error/3 (in-process and CLI); Q7 caller state (a child probe: sentinels before
import and before each call unchanged after import, an ok call, a refused call and a setup failure);
Q6 baseline SHA must resolve to a commit, in-process abstention mutant credited only on a completed
report that fails the predicate, CLI settings hash taken before the children start.

Standard (LL-6/LL-7): --baseline-rev is a full sha (default 087bc36..., main before LL-7/LL-8; pass the
LL-7 landing commit once it exists); temp and cache roots under --out before any library import; a
no-write gate with a sibling-write negative control; source mutants credited only when their target
gates are green unmutated and all turn red. One mutant (abstention_ignored) is in-process: it patches
the loaded miss report, because a source edit of that sha-pinned file is refused by the pin.

Fixtures: tests/test_ll6_inference_bundle.build_fixtures (four families plus a second NN of the same
width) and one planted-signal LightGBM (Step 3 worker with per_seed_sieve_features on; quality =
0.5 * forward_matches + noise), all trained by the real Step 5 CLI on CPU. CPU only. Writes only
under --out, which must lie under <this tree>/data/.

  PYTHONDONTWRITEBYTECODE=1 python3 -u -B tests/test_ll8_attribution_cycle.py --out data/ll78_build/ll8_<stamp>
"""
import argparse
import hashlib
import json
import os
import random
import re
import resource
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tests"))
import test_ll6_inference_bundle as ll6  # noqa: E402

BASELINE_REV = "087bc366fb199dbbe26c1fe918a911158618d358"
AC_REL = "scripts/diagnostics/attribution_cycle.py"
A_NORM_ATOL = 1e-5
RESULTS = []
RUN_INFO = {}


def gate(name, ok, detail=""):
    RESULTS.append({"gate": name, "ok": bool(ok), "detail": str(detail)[:600]})
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {str(detail)[:400]}", flush=True)
    return ok


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


MARK = '__import__("pathlib").Path(__import__("os").environ["LL8_MUTANT_MARK"]).touch()'
# (id, [(old, new, count)], marker, target gates) -- all in scripts/diagnostics/attribution_cycle.py
MUTANTS = [
    ("pin_optional",
     [('    if not (isinstance(want_ck, str) and len(want_ck) == 64 and all(c in "0123456789abcdef" for c in want_ck)):',
       '    if False:', 1),
      ('    if bundle.checkpoint_hash != want_ck:', '    if want_ck and bundle.checkpoint_hash != want_ck:', 1)],
     True, ["LL8.refuses_missing_checkpoint_pin"]),
    ("wrong_pin_ignored",
     [('    if bundle.checkpoint_hash != want_ck:', '    if False:', 1)], True,
     ["LL8.refuses_wrong_checkpoint_same_family"]),
    ("q4_gate_removed",
     [('    if bundle.model_type == "neural_net" and not written_exists and ev is None:', '    if False:', 1)], True,
     ["LL8.g_q4_scaler_refuses_relocated_nn"]),
    # Q3 (Beta X-LL7-LL8 R1)
    ("as_written_without_evidence_allowed",
     [('    if scaler_source == "as_written" and ev is None:', '    if False:', 1)], True,
     ["LL8.q3_as_written_needs_evidence"]),
    ("evidence_scores_unchecked",
     [('        if not gap <= allowed:', '        if False:', 1)], True, ["LL8.q3_wrong_historical_transform_refused"]),
    ("evidence_hash_unchecked",
     [('        if got != ev["scores_sha256"]:', '        if False:', 1)], True,
     ["LL8.q3_wrong_historical_transform_refused"]),
    ("evidence_scaler_state_unchecked",
     [('        if want != scaler_applied:', '        if False:', 1)], True, ["LL8.q3_scaler_state_mismatch_refused"]),
    # Q5
    ("result_alias_allowed",
     [('    if rf.parent == out and rf.name in OUTPUT_NAMES:', '    if False:', 1)], True,
     ["LL8.result_file_distinct_and_never_overwritten"]),
    ("result_overwrites",
     [('os.O_WRONLY | os.O_CREAT | os.O_EXCL', 'os.O_WRONLY | os.O_CREAT | os.O_TRUNC', 1)], True,
     ["LL8.result_file_distinct_and_never_overwritten"]),
    ("publish_error_uncaught",
     [('            _publish(target, res)\n        except Exception as e:',
       '            _publish(target, res)\n        except ZeroDivisionError as e:', 1)], True,
     ["LL8.result_file_distinct_and_never_overwritten"]),
    # Q7
    ("cuda_hidden_at_import",
     [('REPO_ROOT = Path(__file__).resolve().parents[2]\n',
       'os.environ["CUDA_VISIBLE_DEVICES"] = ""\nREPO_ROOT = Path(__file__).resolve().parents[2]\n', 1)], True,
     ["LL8.caller_state_restored"]),
    ("cuda_not_restored",
     [('{k: os.environ.get(k) for k in list(dirs) + list(self.HIDE)}', '{k: os.environ.get(k) for k in list(dirs)}', 1)],
     True, ["LL8.caller_state_restored"]),
    ("sys_path_left",
     [('        sys.path[:] = s["sys_path"]', '        pass', 1)], True, ["LL8.caller_state_restored"]),
    ("root_logging_left",
     [('        root.handlers[:] = s["log_handlers"]\n        root.setLevel(s["log_level"])', '        pass', 1)], True,
     ["LL8.caller_state_restored"]),
    ("torch_rng_left",
     [('        if s["torch_rng"] is not None and "torch" in sys.modules:', '        if False:', 1)], True,
     ["LL8.caller_state_restored"]),
    ("setup_failure_leaks_env",       # both guards removed: directories made inside the env loop, no restore on failure
     [('        for d in dirs.values():                 # all directories first: a failure here has changed nothing\n'
       '            d.mkdir(parents=True, exist_ok=True)\n', '', 1),
      ('            for k, d in dirs.items():\n                os.environ[k] = str(d)\n',
       '            for k, d in dirs.items():\n                os.environ[k] = str(d)\n'
       '                d.mkdir(parents=True, exist_ok=True)\n', 1),
      ('        except BaseException:\n            self._restore()\n            raise\n',
       '        except ZeroDivisionError:\n            raise\n', 1)], True, ["LL8.caller_state_restored"]),
    ("default_path_other_worktree",
     [('MISS_REPORT_DEFAULT = REPO_ROOT / "scripts/diagnostics/e2e_miss_report.py"',
       'MISS_REPORT_DEFAULT = Path.home() / "tfm_worktrees/e2e-vote/scripts/diagnostics/e2e_miss_report.py"', 1)],
     False, ["LL8.miss_report_default_is_this_tree"]),
    ("anorm_skipped",
     [('    if not anorm <= a_allowed:', '    if False:', 1)], True, ["LL8.anorm_refuses_mismatched_forward"]),
    # LL-8 fix2: the allowed gap scales with the scores; without the scale the 2026-11-26 red returns
    ("tolerance_unscaled",
     [('    return atol + rtol * scale, scale', '    return atol, scale', 1)], True,
     ["LL8.anorm_tolerance_scales_with_scores", "LL8.q3_evidence_tolerance_scales_with_scores"]),
    ("nested_bands",
     [('    return [(f"{lo + 1}-{hi}", order[lo:hi]) for lo, hi in edges]',
       '    return [(f"{lo + 1}-{hi}", order[:hi]) for lo, hi in edges]', 1)], True, ["LL8.bands_disjoint"]),
    ("global_counted_as_seed_evidence",
     [('    per_run = [r["name"] for r in per_run_rows]\n',
       '    per_seed = per_seed + [r["name"] for r in per_run_rows if r["name"].startswith("global_")]\n'
       '    per_run_rows = [r for r in per_run_rows if not r["name"].startswith("global_")]\n'
       '    per_run = [r["name"] for r in per_run_rows]\n', 1)], False, ["LL8.per_run_columns_are_context_only"]),
    ("family_check_skipped",
     [('    if model_family is not None and bundle.model_type != model_family:', '    if False:', 1)], True,
     ["LL8.family_and_flag_checks"]),
]


def make_mutant(root, reps, marker):
    src = (REPO / AC_REL).read_text()
    for old, new, count in reps:
        if src.count(old) != count:
            return None, f"anchor {old[:50]!r} found {src.count(old)} times, want {count}"
    if marker:
        lines, done = [], False
        for line in src.split("\n"):
            if reps[0][0].split("\n")[0] in line and not done:
                lines.append(line[:len(line) - len(line.lstrip())] + MARK)
                done = True
            lines.append(line)
        src = "\n".join(lines)
    for old, new, count in reps:
        src = src.replace(old, new)
    f = root / "code" / AC_REL
    f.parent.mkdir(parents=True)
    f.write_text(src)
    return f, f"sha {sha(REPO / AC_REL)[:12]} -> {sha(f)[:12]}"


def run_mutants(out, jobs, models_src, green, baseline_rev, boundary):
    def one(m):
        mid, reps, marker, targets = m
        root = out / "mutants" / mid
        unmut = {t: green.get(t) for t in targets}
        if not all(v is True for v in unmut.values()):
            return {"id": mid, "killed": False, "targets": targets, "unmutated": unmut,
                    "detail": "a target gate is not green in the unmutated run: no credit"}
        f, how = make_mutant(root, reps, marker)
        if f is None:
            return {"id": mid, "killed": False, "detail": how}
        mark = root / "mark"
        p = subprocess.run([sys.executable, "-u", "-B", str(Path(__file__).resolve()), "--out", str(root / "run"),
                            "--ac-file", str(f), "--models-from", str(models_src), "--no-mutants",
                            "--only", ",".join(targets), "--baseline-rev", baseline_rev, "--boundary", str(boundary)],
                           capture_output=True, text=True,
                           env=dict(os.environ, LL8_MUTANT_MARK=str(mark), CUDA_VISIBLE_DEVICES="",
                                    PYTHONDONTWRITEBYTECODE="1"))
        (root / "run.log").write_text(p.stdout + p.stderr)
        try:
            g = {r["gate"]: r["ok"] for r in json.load(open(root / "run" / "selftest_summary.json"))["gates"]}
        except Exception as e:
            return {"id": mid, "killed": False, "detail": f"no summary ({e}); rc {p.returncode}"}
        red = [t for t in targets if g.get(t) is False]
        killed = len(red) == len(targets) and (mark.exists() or not marker)
        return {"id": mid, "kind": "source", "killed": killed, "targets_red": red, "targets": targets,
                "unmutated": unmut, "marker": ("hit" if mark.exists() else "absent") if marker else "n/a",
                "applied": how}
    res = []
    with ThreadPoolExecutor(jobs) as ex:
        for r in ex.map(one, MUTANTS):
            res.append(r)
            print(f"[{'PASS' if r['killed'] else 'FAIL'}] MUTANT LL8.{r['id']} killed={r['killed']} "
                  f"red={r.get('targets_red')} marker={r.get('marker')} {r.get('applied', r.get('detail'))}", flush=True)
    return res


def baseline_lacks_ac(rev):
    """Beta X-LL7-LL8 R1 Q6: the baseline must resolve to a commit before path absence means anything."""
    c = subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{rev}^{{commit}}"], capture_output=True)
    if c.returncode != 0:
        return False, f"{rev[:12]} does not resolve to a commit"
    absent = ll6.git_show(rev, AC_REL) is None
    return absent, f"{rev[:12]} is a commit; {AC_REL} absent there: {absent}"


def step6_hash(y):
    """Independent copy of the vote harness's RankOnce.scores_sha256 (dtype name + raw bytes)."""
    import numpy as np
    y = np.ascontiguousarray(np.asarray(y))
    return hashlib.sha256(str(y.dtype).encode() + y.tobytes()).hexdigest()


def abstention_credit(calls, res, rep, predicate):
    """Q6: credit only when the hook ran AND a completed report fails the sparse-cohort predicate."""
    return calls > 0 and res is not None and res.get("status") == "ok" and rep is not None and not predicate(rep)


# Q7 probe: a fresh process; sentinels set before import and before each call (Beta X-LL7-LL8 R1 Q7)
CALLER_PROBE = r'''
import hashlib, importlib.util, json, logging, os, sys, tempfile
from pathlib import Path
ac_file, repo, sp_ok, sp_ref, work = sys.argv[1:6]
work = Path(work)
(work / "probe_tmp").mkdir(parents=True, exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "ll8-sentinel-import"
os.environ["TMPDIR"] = str(work / "probe_tmp")
os.environ["XDG_CACHE_HOME"] = str(work / "probe_tmp" / "ll8-sentinel-cache")
os.environ.pop("MPLCONFIGDIR", None)
os.environ.pop("CUDA_CACHE_PATH", None)
import numpy, torch
torch.manual_seed(1234)
KEYS = ("CUDA_VISIBLE_DEVICES", "TMPDIR", "XDG_CACHE_HOME", "MPLCONFIGDIR", "CUDA_CACHE_PATH")
def state():
    root = logging.getLogger()
    return {"env": {k: os.environ.get(k, "<absent>") for k in KEYS}, "tempdir": tempfile.tempdir,
            "sys_path": list(sys.path), "root_handlers": [type(h).__name__ for h in root.handlers],
            "root_level": root.level,
            "torch_rng": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()}
def diff(a, b):
    return {k: [str(a[k])[:200], str(b[k])[:200]] for k in a if a[k] != b[k]}
res = {}
s0 = state()
spec = importlib.util.spec_from_file_location("ac_probe", ac_file)
ac = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ac)
res["import"] = diff(s0, state())
ac.REPO_ROOT = Path(repo)
miss = str(Path(repo) / "scripts/diagnostics/e2e_miss_report.py")
os.environ["CUDA_VISIBLE_DEVICES"] = "ll8-sentinel-call"
seen = []
orig = ac._cycle
def spy(*a, **k):
    seen.append(os.environ.get("CUDA_VISIBLE_DEVICES"))
    return orig(*a, **k)
ac._cycle = spy
calls = {}
for tag, sp in (("ok", sp_ok), ("refused", sp_ref)):
    s1 = state()
    r = ac.run_attribution_cycle(sp, str(work / ("probe_" + tag)), miss_report=miss)
    calls[tag] = {"status": r["status"], "message": str(r.get("message"))[:200], "changed": diff(s1, state())}
keys = ac._CallerScope.KEYS
ac._CallerScope.KEYS = dict(keys, CUDA_CACHE_PATH="bad\0dir")
s1 = state()
try:
    r = ac.run_attribution_cycle(sp_ok, str(work / "probe_setup"), miss_report=miss)
    calls["setup_failure"] = {"status": r["status"], "message": str(r.get("message"))[:200], "changed": diff(s1, state())}
except Exception as e:
    calls["setup_failure"] = {"status": "raised", "message": repr(e)[:200], "changed": diff(s1, state())}
finally:
    ac._CallerScope.KEYS = keys
res["calls"] = calls
res["cuda_seen_in_cycle"] = seen
print("PROBE " + json.dumps(res), flush=True)
'''


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def planted_fixture(out):
    """600 Step 3 rows (per_seed_sieve_features on), quality planted on forward_matches; one LightGBM."""
    import test_ll_feature_settings as t1
    fx = out / "fx_planted"
    t1.make_fixture(fx, seed=31, n=600)
    o = out / "step3_planted.json"
    subprocess.run([sys.executable, "-B", str(REPO / "tests/ll_step3_runner.py"), "--code-dir", str(REPO),
                    "--repo-root", str(REPO), "--fixture", str(fx), "--settings",
                    json.dumps({"per_seed_sieve_features": "on"}), "--out", str(o)], check=True, capture_output=True)
    r = random.Random(31)
    rows = [{"seed": x["seed"], "features": x["features"], "holdout_hits": x["holdout_hits"],
             "holdout_quality": 0.5 * x["features"]["forward_matches"] + 0.002 * r.random(), "score": x["score"]}
            for x in json.load(open(o))["results"]]
    surv = out / "survivors_planted.json"
    json.dump(rows, open(surv, "w"))
    R = out / "models_planted_lgbm"
    R.mkdir()
    shutil.copy(REPO / "train_single_trial.py", R / "train_single_trial.py")   # TODO-E2E-24, as LL-6
    env = dict(os.environ, PYTHONPATH=str(REPO), CUDA_VISIBLE_DEVICES="", S88_FORCE_LGBM_CPU="1",
               PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([sys.executable, "-u", "-B", str(REPO / "meta_prediction_optimizer_anti_overfit.py"),
                        "--survivors", str(surv), "--lottery-data", str(fx / "train.json"), "--trials", "1",
                        "--k-folds", "2", "--model-type", "lightgbm", "--device", "cpu",
                        "--storage", f"sqlite:///{R}/optuna.db", "--study-name", "ll8_planted",
                        "--output-dir", str(R / "models/reinforcement")], cwd=R, env=env, capture_output=True, text=True)
    (R / "run.log").write_text(p.stdout + p.stderr)
    if p.returncode != 0:
        raise RuntimeError(f"planted Step 5 failed rc={p.returncode}; see {R}/run.log")
    return surv, R / "models/reinforcement"


def relocate_nn(out, src_dir):
    """Copy an NN models dir and point the sidecar's WRITTEN checkpoint_path at a path that does not
    exist, as after a move (Beta X-LL6 R1 Q4). The loader then resolves models_dir/<basename>."""
    d = out / "models_nn_relocated"
    if d.exists():
        return d
    shutil.copytree(src_dir, d)
    meta = json.load(open(d / "best_model.meta.json"))
    meta["checkpoint_path"] = str(out / "moved_away" / Path(meta["checkpoint_path"]).name)
    json.dump(meta, open(d / "best_model.meta.json", "w"), indent=2)
    return d


def write_votes(out, tag, rows, outc_fn, n_draws=330, t_start=20000, seed=11):
    import numpy as np
    rr = np.random.default_rng(seed)
    V = rr.integers(0, 1000, size=(len(rows), n_draws)).astype(np.int16)
    outc = outc_fn(V, rr, n_draws)
    cidx = np.arange(t_start, t_start + n_draws)
    votes = out / f"votes_{tag}.npz"
    np.savez(votes, V=V, seeds=np.array([r["seed"] for r in rows], dtype=np.int64), combined_index=cidx)
    gd = out / f"grade_{tag}"
    gd.mkdir()
    lines = [json.dumps({"combined_index": int(t), "outcome": int(o), "status": "graded",
                         "session": "midday" if t % 2 else "evening", "date": "2027-01-01"}) for t, o in zip(cidx, outc)]
    (gd / "per_draw.jsonl").write_text("\n".join(lines) + "\n")
    (gd / "SHA256SUMS").write_text(f"{sha(gd / 'per_draw.jsonl')}  per_draw.jsonl\n")
    return votes, gd, t_start


def make_spec(out, tag, surv, models_dir, votes_path, gd, t_start, **over):
    spec = {"label": "SELF-TEST FIXTURE", "survivors": {"path": str(surv), "sha256": sha(surv)},
            "models_dir": str(models_dir), "votes": {"path": str(votes_path), "sha256": sha(votes_path)},
            "grade_dirs": [str(gd)],
            "blocks": {"A": [t_start, t_start + 160], "B": [t_start + 160, t_start + 320],
                       "sparse": [t_start + 320, t_start + 326]},
            "sessions": ["midday", "evening"], "pool_sizes": [20, 100, 300],
            "per_run_prefixes": ["global_"], "abstain": {"min_pairs": 20, "min_draws": 10, "min_survivors": 5},
            "band_min_survivors": 10, "bootstrap": {"B": 400}, "seed": 7,
            "attribution": {"ig_steps": 64}, "a_norm_atol": A_NORM_ATOL}
    for k, v in over.items():
        if v is None:
            spec.pop(k, None)
        else:
            spec[k] = v
    p = out / f"spec_{tag}.json"
    p.write_text(json.dumps(spec, indent=1))
    return p


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline-rev", default=BASELINE_REV)
    ap.add_argument("--ac-file", default=None, help="mutant runs only: the attribution_cycle.py under test")
    ap.add_argument("--models-from", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--no-mutants", action="store_true")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--boundary", default=None)
    ap.add_argument("--clean-on-pass", action="store_true")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if not str(out).startswith(str(REPO / "data") + "/"):
        raise SystemExit("--out must lie under <this tree>/data/")
    if not re.fullmatch(r"[0-9a-f]{40}", a.baseline_rev):
        raise SystemExit("--baseline-rev must be a full 40-hex sha")
    out.mkdir(parents=True)
    RUN_INFO["baseline_rev"] = a.baseline_rev
    RUN_INFO["env_roots"] = ll6.redirect_env_roots(out)
    import numpy as np
    boundary = Path(a.boundary).resolve() if a.boundary else out
    if out != boundary and not str(out).startswith(str(boundary) + "/"):
        raise SystemExit("--out must lie inside --boundary")
    if not str(boundary).startswith(str(REPO / "data") + "/"):
        raise SystemExit("--boundary must lie under <this tree>/data/")
    only = set(a.only.split(",")) if a.only else None
    want = (lambda g: only is None or g in only)
    mutant_run = a.ac_file is not None
    sys.path.insert(0, str(REPO))
    t0 = time.time()
    guard_before = ll6.boundary_snapshot(boundary)

    # ---- baseline precondition --------------------------------------------------
    absent, why = baseline_lacks_ac(a.baseline_rev)
    ctl_absent, ctl_why = baseline_lacks_ac("0" * 40)          # control: an unknown sha must NOT pass
    psa_now = (REPO / "per_survivor_attribution.py").read_text()
    pre_ok = (absent and not ctl_absent and "per_survivor_attribution_signed_matrix" in psa_now
              and (REPO / AC_REL).exists())
    gate("LL8.baseline_precondition", pre_ok,
         f"{why}; control {ctl_why} -> passes: {ctl_absent}; this tree has LL-7's signed functions: "
         f"{'per_survivor_attribution_signed_matrix' in psa_now}")
    if not pre_ok:
        return finish(out, t0, [], guard_before, boundary, a)

    # ---- module under test --------------------------------------------------------
    ac_file = Path(a.ac_file).resolve() if a.ac_file else REPO / AC_REL
    ac = ll6.load_as("attribution_cycle_under_test", ac_file)
    own_root = ac_file.parents[2]
    try:
        default_rel = Path(ac.MISS_REPORT_DEFAULT).relative_to(own_root)
    except ValueError:
        default_rel = None
    if mutant_run:      # a mutant copy lives outside the tree: bind it to this tree (paths it computes from its own location)
        ac.REPO_ROOT = REPO
        if default_rel is not None:
            ac.MISS_REPORT_DEFAULT = REPO / default_rel
    if want("LL8.miss_report_default_is_this_tree"):
        dflt = Path(ac.MISS_REPORT_DEFAULT)
        gate("LL8.miss_report_default_is_this_tree",
             default_rel == Path("scripts/diagnostics/e2e_miss_report.py") and dflt.exists()
             and sha(dflt) == ac.MISS_REPORT_SHA256 and ac.parse_args(["--spec", "s", "--out-dir", "o"]).miss_report == str(dflt),
             f"default {dflt} relative to the script's tree: {default_rel}; sha256 equals the pin: "
             f"{dflt.exists() and sha(dflt) == ac.MISS_REPORT_SHA256}")

    # ---- fixtures -------------------------------------------------------------------
    t_fix = time.time()
    if a.models_from:
        src = Path(a.models_from).resolve()
        surv, dirs = src / "survivors.json", {f: src / f"models_{f}" / "models/reinforcement"
                                               for f in ll6.FAMILIES + ("neural_net_alt",)}
        surv_p, dir_p = src / "survivors_planted.json", src / "models_planted_lgbm" / "models/reinforcement"
        fx_root = src
    else:
        box = {}
        th = threading.Thread(target=lambda: box.update(p=planted_fixture(out)))
        th.start()
        surv, _hist, dirs = ll6.build_fixtures(out)
        th.join()
        if "p" not in box:
            raise RuntimeError("planted fixture failed")
        surv_p, dir_p = box["p"]
        fx_root = out
    t_fix = time.time() - t_fix
    rows_p = json.load(open(surv_p))
    rows = json.load(open(surv))
    import per_survivor_attribution as psa
    import models.inference_bundle as ib
    bP = ib.load_inference_bundle(str(dir_p))
    ck = {k: ib.load_inference_bundle(str(v)).checkpoint_hash for k, v in dirs.items()}
    ck["planted"] = bP.checkpoint_hash

    # planted votes: outcomes copy the vote of a top-20 survivor that leans hardest on forward_matches
    XP = bP.matrix([r["features"] for r in rows_p])
    orderP = np.argsort(bP.predict(XP))[::-1]
    MP = psa.per_survivor_attribution_signed_matrix(bP.attribution_model, "lightgbm", bP.transform(XP), bP.feature_names)
    jf = bP.feature_names.index("forward_matches")
    top20 = orderP[:20]
    a_fm = MP["signed"][top20, jf]
    strong = top20[a_fm >= np.median(a_fm)]

    def planted_outc(V, rr, n):
        return np.array([V[strong[rr.integers(len(strong))], j] if rr.random() < 0.5 else rr.integers(0, 1000)
                         for j in range(n)])
    w = out / "work"
    w.mkdir()
    votes_p, gd_p, t_s = write_votes(w, "planted", rows_p, planted_outc)
    votes_n, gd_n, _ = write_votes(w, "nn", rows, lambda V, rr, n: rr.integers(0, 1000, size=n))
    sp_main = make_spec(w, "main", surv_p, dir_p, votes_p, gd_p, t_s, expect_checkpoint_sha256=ck["planted"])

    def cycle(tag, spec_path, **kw):
        od = w / f"cyc_{tag}"
        res = ac.run_attribution_cycle(str(spec_path), str(od), **kw)
        rep = prio = rrec = None
        if res["status"] == "ok":
            rep, prio = json.load(open(od / "attribution_cycle.json")), json.load(open(od / "extraction_priorities.json"))
            rrec = json.load(open(od / "run_record.json"))
        return res, rep, prio, rrec, od

    def anorm_recorded_ok(rep, rtol):
        # the report records gap, atol, rtol, scale and allowed; allowed = atol + rtol * scale; gap within it
        a = (rep or {}).get("attribution", {})
        try:
            return (a["a_norm_atol"] == A_NORM_ATOL and a["a_norm_rtol"] == rtol and a["a_norm_score_scale"] > 0
                    and abs(a["a_norm_allowed"] - (a["a_norm_atol"] + a["a_norm_rtol"] * a["a_norm_score_scale"])) < 1e-15
                    and 0 <= a["a_norm_max_abs_diff"] <= a["a_norm_allowed"])
        except (KeyError, TypeError):
            return False

    def g_planted(rep, prio):
        top = prio["per_seed_groups_ranked"][0]["group"] if prio["per_seed_groups_ranked"] else None
        rec = [r for r in rep["recurring_C1b"] if r["feature"] == "forward_matches" and r["sign"] == 1]
        ds = rec[0]["draw_sets"] if rec else []
        return (top == "per_seed_sieve" and len(ds) >= 2 and len({d.split("/")[0] for d in ds}) == 2), \
            f"top group {top}; forward_matches recurring in draw sets {ds}"

    def g_context(rep, prio):
        pr = {r["name"] for r in rep["namespaces"]["per_run"]}
        glob = {n for n in bP.feature_names if n.startswith("global_")}
        in_c = set()
        for cell in rep["contrasts"].values():
            for cc in cell.get("cohorts", {}).values():
                for k in ("C1", "C1b"):
                    in_c |= {f["feature"] for f in cc[k].get("features", [])}
        ctx = {g["group"] for g in prio["per_run_context_groups"]}
        return (bool(glob) and glob <= pr and not (in_c & pr) and "global" in ctx
                and "global" not in {g["group"] for g in prio["per_seed_groups_ranked"]}), \
            f"{len(glob)} global_ columns, {len(pr)} per-run columns, none in contrasts; context groups {sorted(ctx)}"

    def g_sparse(rep):
        cell = rep["contrasts"].get("sparse/midday", {})
        cc = cell.get("cohorts", {}).get("20", {})
        return (cell.get("draws") == 3 and cc.get("C1b", {}).get("abstained") is True
                and cc.get("C1", {}).get("abstained") is True), f"sparse/midday draws={cell.get('draws')}"

    def g_bands(rep):
        sizes = [b["survivors"] for b in rep["bands"]]
        labels = [b["band"] for b in rep["bands"]]
        return sizes == [20, 80, 200, 300] and labels == ["1-20", "21-100", "101-300", "301-600"], f"{labels} {sizes}"

    main_gates = ("LL8.runs_and_A-NORM_inside", "LL8.planted_per_seed_signal_ranks_first",
                  "LL8.per_run_columns_are_context_only", "LL8.sparse_cohort_abstains", "LL8.bands_disjoint",
                  "LL8.priorities_carry_advisory_and_LL3_variants", "LL8.run_record_binds_model")
    if any(want(g) for g in main_gates):
        res, rep, prio, rrec, od = cycle("main", sp_main, model_family="lightgbm", flag_id="lightgbm@2026-11-26T00:00:00+00:00")
        ok = res["status"] == "ok"
        if want("LL8.runs_and_A-NORM_inside"):
            outs_ok = ok and all(sha(od / n) == h for n, h in res["outputs"].items())
            rj = json.load(open(od / "result.json")) if (od / "result.json").exists() else {}
            gate("LL8.runs_and_A-NORM_inside", ok and anorm_recorded_ok(rep, ac.A_NORM_RTOL_DEFAULT)
                 and outs_ok and rj.get("status") == "ok" and rj.get("exit_code") == 0,
                 f"status {res['status']} {res.get('message')}; a_norm "
                 f"{rep and {k: v for k, v in rep['attribution'].items() if k.startswith('a_norm_') and k != 'a_norm_scope'}}; result.json outputs sha match files: {outs_ok}")
        if want("LL8.planted_per_seed_signal_ranks_first"):
            gate("LL8.planted_per_seed_signal_ranks_first", *(g_planted(rep, prio) if ok else (False, res)))
        if want("LL8.per_run_columns_are_context_only"):
            gate("LL8.per_run_columns_are_context_only", *(g_context(rep, prio) if ok else (False, res)))
        if want("LL8.sparse_cohort_abstains"):
            gate("LL8.sparse_cohort_abstains", *(g_sparse(rep) if ok else (False, res)))
        if want("LL8.bands_disjoint"):
            gate("LL8.bands_disjoint", *(g_bands(rep) if ok else (False, res)))
        if want("LL8.priorities_carry_advisory_and_LL3_variants"):
            gate("LL8.priorities_carry_advisory_and_LL3_variants",
                 ok and prio["advisory"].startswith("ADVISORY") and rep["advisory"].startswith("ADVISORY")
                 and len(prio["per_seed_groups_ranked"]) > 0 and all(
                     g["test_with"]["step5_variant"].startswith("--exclude-feature-groups ")
                     for g in prio["per_seed_groups_ranked"]), (prio or {}).get("advisory", "")[:40])
        if want("LL8.run_record_binds_model"):
            mb = (rrec or {}).get("model_binding", {})
            side = json.load(open(dir_p / "best_model.meta.json"))
            gate("LL8.run_record_binds_model",
                 ok and mb.get("checkpoint_sha256") == ck["planted"] == mb.get("expect_checkpoint_sha256")
                 and mb.get("sidecar_written_checkpoint_path") == side["checkpoint_path"]
                 and mb.get("feature_schema_hash") == side["feature_schema"].get("feature_schema_hash")
                 and mb.get("feature_schema_hash") is not None and mb.get("scaler_source") == "resolved"
                 and mb.get("step6_match", {}).get("checked") is False
                 and rep["step6_match"]["checked"] is False and "NOT checked" in rep["scaler_note"]
                 and rrec["run_for"] == {"model_family": "lightgbm", "flag_id": "lightgbm@2026-11-26T00:00:00+00:00"}
                 and rep["run_for"] == rrec["run_for"] and res["flag_id"] == rrec["run_for"]["flag_id"],
                 f"model_binding {mb}; run_for {(rrec or {}).get('run_for')}")

    # ---- refusals ----------------------------------------------------------------------
    def refused(tag, spec_path, needle=None, **kw):
        res, *_r, od = cycle(tag, spec_path, **kw)
        files = sorted(p.name for p in od.iterdir() if p.name != "_env") if od.exists() else None
        ok = (res["status"] == "refused" and res["exit_code"] == 2 and res["error_class"] == "CycleError"
              and (needle is None or needle in (res["message"] or "")) and files in (["result.json"], None))
        return ok, f"{tag}: {res['status']} {str(res.get('message'))[:120]} files={files}"

    if want("LL8.refusals_inputs"):
        det = []
        bad_v = make_spec(w, "bad_votes", surv_p, dir_p, votes_p, gd_p, t_s,
                          expect_checkpoint_sha256=ck["planted"], votes={"path": str(votes_p), "sha256": "0" * 64})
        det.append(refused("votes_sha", bad_v, "votes sha256"))
        det.append(refused("miss_sha", sp_main, "miss report", miss_report_sha256="0" * 64))
        r_out = ac.run_attribution_cycle(str(sp_main), "/tmp/ll8_never_written")
        det.append((r_out["status"] == "refused" and r_out["result_written"] is None
                    and not Path("/tmp/ll8_never_written").exists(), f"outside data: {r_out['message']}"))
        ex = w / "cyc_exists"
        ex.mkdir()
        (ex / "keep.txt").write_text("x")
        r_ex = ac.run_attribution_cycle(str(sp_main), str(ex))
        det.append((r_ex["status"] == "refused" and sorted(p.name for p in ex.iterdir()) == ["keep.txt"],
                    f"exists: {r_ex['message']}"))
        gate("LL8.refusals_inputs", all(o for o, _ in det), [d for _, d in det])
    sp_nopin = make_spec(w, "nopin", surv_p, dir_p, votes_p, gd_p, t_s, expect_checkpoint_sha256=None)
    if want("LL8.refuses_missing_checkpoint_pin"):
        o1, d1 = refused("nopin", sp_nopin, "expect_checkpoint_sha256 is required")
        sp_short = make_spec(w, "shortpin", surv_p, dir_p, votes_p, gd_p, t_s, expect_checkpoint_sha256=ck["planted"][:12])
        o2, d2 = refused("shortpin", sp_short, "expect_checkpoint_sha256 is required")
        gate("LL8.refuses_missing_checkpoint_pin", o1 and o2, [d1, d2])
    nn_gates = ("LL8.refuses_wrong_checkpoint_same_family", "LL8.family_and_flag_checks",
                "LL8.anorm_refuses_mismatched_forward", "LL8.anorm_tolerance_scales_with_scores")
    if any(want(g) for g in nn_gates + ("LL8.caller_state_restored",)):
        sp_nn = make_spec(w, "nn", surv, dirs["neural_net"], votes_n, gd_n, t_s, expect_checkpoint_sha256=ck["neural_net"])
    if want("LL8.refuses_wrong_checkpoint_same_family"):
        b1, b2 = ib.load_inference_bundle(str(dirs["neural_net"])), ib.load_inference_bundle(str(dirs["neural_net_alt"]))
        same = b1.model_type == b2.model_type and b1.feature_names == b2.feature_names and ck["neural_net"] != ck["neural_net_alt"]
        sp_wrong = make_spec(w, "wrongck", surv, dirs["neural_net_alt"], votes_n, gd_n, t_s,
                             expect_checkpoint_sha256=ck["neural_net"])
        o1, d1 = refused("wrongck", sp_wrong, "!= declared")
        res_c, *_x = cycle("nn_control", sp_nn, model_family="neural_net")
        gate("LL8.refuses_wrong_checkpoint_same_family", o1 and same and res_c["status"] == "ok",
             f"{d1}; non-vacuity: same family and names, different checkpoint: {same}; control (right pin) -> "
             f"{res_c['status']} {res_c.get('message')}")
    q3_gates = ("LL8.g_q4_scaler_refuses_relocated_nn", "LL8.g_q4_as_written_runs_and_labels",
                "LL8.q3_wrong_historical_transform_refused", "LL8.q3_scaler_state_mismatch_refused",
                "LL8.q3_as_written_needs_evidence", "LL8.q3_resolved_matches_step6_evidence",
                "LL8.q3_evidence_tolerance_scales_with_scores")
    if any(want(g) for g in q3_gates):
        import prediction_generator as pg
        seeds_n = np.array([int(r["seed"]) for r in rows], dtype=np.int64)
        # Beta X-LL7-LL8 R1 Q3 fixture: Step 6 scores FIRST, where the model was written (scaler found) ...
        y6_first, _g = ll6.step6_in_process(pg, dirs["neural_net"], rows)
        ev_first = w / "step6_scores_first.npz"
        np.savez(ev_first, scores=y6_first, seeds=seeds_n)
        rel = relocate_nn(out, dirs["neural_net"])          # ... the model is relocated SECOND
        ck_rel = ib.load_inference_bundle(str(rel)).checkpoint_hash
        b_res = ib.load_inference_bundle(str(rel), scaler_source="resolved")
        b_aw = ib.load_inference_bundle(str(rel), scaler_source="as_written")
        Xn = b_res.matrix([r["features"] for r in rows])
        y6, _gen = ll6.step6_in_process(pg, rel, rows)      # Step 6 run AFTER relocation: unscaled
        ev_after = w / "step6_scores_after.npz"
        np.savez(ev_after, scores=y6, seeds=seeds_n)
        p_res, p_aw = b_res.predict(Xn), b_aw.predict(Xn)
        gap_res = float(np.max(np.abs(y6 - p_res)))
        gap_aw = float(np.max(np.abs(y6 - p_aw)))
        gap_first_aw = float(np.max(np.abs(y6_first - p_aw)))
        gap_first_res = float(np.max(np.abs(y6_first - p_res)))
        M_aw = psa.per_survivor_attribution_signed_matrix(
            b_aw.attribution_model, b_aw.model_type, b_aw.transform(Xn), b_aw.feature_names,
            expected_feature_count=b_aw.feature_count, y_mean=b_aw.y_mean, y_std=b_aw.y_std, ig_steps=64)
        anorm_aw = float(np.max(np.abs(M_aw["prediction"] - p_aw)))
        allow_aw = ac.allowed_gap(A_NORM_ATOL, ac.A_NORM_RTOL_DEFAULT, p_aw, "A-NORM")[0]
        allow6 = ac.allowed_gap(ac.STEP6_ATOL_DEFAULT, ac.STEP6_RTOL_DEFAULT, y6, "step6")[0]
        allow6_first = ac.allowed_gap(ac.STEP6_ATOL_DEFAULT, ac.STEP6_RTOL_DEFAULT, y6_first, "step6")[0]
        RUN_INFO["q4_measured"] = {"step6_vs_resolved_max_abs": gap_res, "step6_vs_as_written_max_abs": gap_aw,
                                   "step6_first_vs_as_written_max_abs": gap_first_aw,
                                   "step6_first_vs_resolved_max_abs": gap_first_res,
                                   "as_written_internal_a_norm": anorm_aw, "as_written_a_norm_allowed": allow_aw,
                                   "step6_after_allowed": allow6, "step6_first_allowed": allow6_first,
                                   "hash_first_equals_resolved": step6_hash(y6_first) == step6_hash(p_res),
                                   "hash_after_equals_as_written": step6_hash(y6) == step6_hash(p_aw)}

        def ev(path, **extra):
            return dict({"scores": {"path": str(path), "sha256": sha(path)}}, **extra)

        def q3spec(tag, **kw):
            return make_spec(w, tag, surv, rel, votes_n, gd_n, t_s, expect_checkpoint_sha256=ck_rel, **kw)
        if want("LL8.g_q4_scaler_refuses_relocated_nn"):
            o1, d1 = refused("q4_default", q3spec("q4"), "G-Q4-SCALER")
            o2, d2 = refused("q4_resolved", q3spec("q4r", scaler_source="resolved"), "G-Q4-SCALER")
            gate("LL8.g_q4_scaler_refuses_relocated_nn", o1 and o2 and gap_res > allow6 and gap_aw <= allow6,
                 f"{d1}; {d2}; non-vacuity: Step 6 on the relocated dir differs from the resolved-scaler bundle by "
                 f"{gap_res:.3e} and from the as_written bundle by {gap_aw:.1e}")
        if want("LL8.g_q4_as_written_runs_and_labels"):
            sp_aw = q3spec("q4aw", scaler_source="as_written",
                           step6_evidence=ev(ev_after, scaler_applied=False, scores_sha256=step6_hash(y6),
                                             source="Step 6 run after relocation (fixture)"))
            res, rep, prio, rrec, od = cycle("q4_as_written", sp_aw)
            sm = (rep or {}).get("step6_match", {})
            gate("LL8.g_q4_as_written_runs_and_labels",
                 res["status"] == "ok" and sm.get("checked") is True and len(sm.get("items", [])) == 3
                 and sm.get("scaler_applied_this_run") is False
                 and "Step 6 match checked" in rep["scaler_note"] and prio["step6_match"] == sm
                 and rrec["model_binding"]["scaler_source"] == "as_written"
                 and rrec["model_binding"]["step6_match"] == sm
                 and rrec["model_binding"]["sidecar_written_checkpoint_exists"] is False
                 and rep["model"]["normalize_features"] is False and anorm_recorded_ok(rep, ac.A_NORM_RTOL_DEFAULT),
                 f"status {res['status']} {res.get('message')}; step6_match {sm}")
        if want("LL8.q3_wrong_historical_transform_refused"):
            det = [refused("q3_aw_scores", q3spec("q3aws", scaler_source="as_written", step6_evidence=ev(ev_first)),
                           "Step 6 evidence"),
                   refused("q3_aw_hash", q3spec("q3awh", scaler_source="as_written",
                                                step6_evidence={"scores_sha256": step6_hash(y6_first)}),
                           "Step 6 evidence"),
                   refused("q3_res_wrong", q3spec("q3rw", scaler_source="resolved", step6_evidence=ev(ev_after)),
                           "Step 6 evidence")]
            nonvac = ck_rel == ck["neural_net"] and anorm_aw <= allow_aw and gap_first_aw > allow6_first
            gate("LL8.q3_wrong_historical_transform_refused", all(o for o, _ in det) and nonvac,
                 [d for _, d in det] + [f"non-vacuity: relocated pin == original pin {ck_rel == ck['neural_net']}; "
                                        f"as_written internal A-NORM {anorm_aw:.1e} <= allowed {allow_aw:.1e} (green); as_written vs the scores "
                                        f"Step 6 made before relocation {gap_first_aw:.3e}"])
        if want("LL8.q3_scaler_state_mismatch_refused"):
            o1, d1 = refused("q3_state", q3spec("q3st", scaler_source="resolved",
                                                step6_evidence=ev(ev_first, scaler_applied=False)), "scaler_applied")
            gate("LL8.q3_scaler_state_mismatch_refused", o1 and gap_first_res <= allow6_first,
                 f"{d1}; non-vacuity: the scores alone match ({gap_first_res:.1e}), only the scaler state differs")
        if want("LL8.q3_as_written_needs_evidence"):
            o1, d1 = refused("q3_aw_none", q3spec("q3awn", scaler_source="as_written"), "needs step6_evidence")
            gate("LL8.q3_as_written_needs_evidence", o1 and gap_aw <= allow6,
                 f"{d1}; non-vacuity: this as_written run WOULD match the unscaled Step 6 ({gap_aw:.1e}) but nothing shows it")
        if want("LL8.q3_resolved_matches_step6_evidence"):
            sp_m = q3spec("q3m", scaler_source="resolved",
                          step6_evidence=ev(ev_first, scaler_applied=True, scores_sha256=step6_hash(y6_first),
                                            source="Step 6 before relocation (fixture)"))
            res, rep, prio, rrec, od = cycle("q3_match", sp_m, model_family="neural_net")
            sm = (rep or {}).get("step6_match", {})
            gate("LL8.q3_resolved_matches_step6_evidence",
                 res["status"] == "ok" and sm.get("checked") is True and len(sm.get("items", [])) == 3
                 and sm.get("scaler_applied_this_run") is True and all(sha(od / n) == h for n, h in res["outputs"].items()),
                 f"status {res['status']} {str(res.get('message'))[:160]}; step6_match {sm}")
        if want("LL8.q3_evidence_tolerance_scales_with_scores"):
            # saved Step 6 scores = the real ones + an offset above atol but inside atol + rtol * max|score|
            # (rtol declared 1e-2 so the offset dwarfs float noise): accepted; twice the allowed gap: refused
            rt = 1e-2
            S6 = float(np.max(np.abs(y6_first)))
            d_in = ac.STEP6_ATOL_DEFAULT + 0.5 * rt * S6
            d_out = 2.0 * (ac.STEP6_ATOL_DEFAULT + rt * S6)
            det = []
            for tg, d in (("in", d_in), ("out", d_out)):
                fp = w / f"step6_scores_offset_{tg}.npz"
                np.savez(fp, scores=np.asarray(y6_first, dtype=np.float64) + d, seeds=seeds_n)
                det.append(q3spec(f"q3tol{tg}", scaler_source="resolved",
                                  step6_evidence={"scores": {"path": str(fp), "sha256": sha(fp), "rtol": rt}}))
            res, rep, prio, rrec, od = cycle("q3_tol_in", det[0])
            it = [i for i in ((rep or {}).get("step6_match") or {}).get("items", []) if "max_abs_diff" in i]
            in_ok = (res["status"] == "ok" and len(it) == 1 and it[0]["rtol"] == rt
                     and it[0]["atol"] == ac.STEP6_ATOL_DEFAULT and it[0]["max_abs_diff"] > ac.STEP6_ATOL_DEFAULT
                     and it[0]["max_abs_diff"] <= it[0]["allowed_gap"]
                     and abs(it[0]["allowed_gap"] - (it[0]["atol"] + rt * it[0]["score_scale"])) < 1e-12)
            o2, d2 = refused("q3_tol_out", det[1], "> allowed")
            gate("LL8.q3_evidence_tolerance_scales_with_scores", in_ok and o2 and d_in > ac.STEP6_ATOL_DEFAULT,
                 [f"offset {d_in:.3e} (> atol {ac.STEP6_ATOL_DEFAULT}) inside atol + {rt} x {S6:.3e}: "
                  f"{res['status']} {str(res.get('message'))[:120]} item {it}", f"offset {d_out:.3e}: {d2}"])
    if want("LL8.anorm_tolerance_scales_with_scores"):
        # an injected gap above atol but inside atol + rtol * max|score| is accepted; twice the allowed gap is
        # refused; the report records gap, atol, rtol, scale and allowed (rtol declared 1e-2: offset >> float noise)
        orig = psa.per_survivor_attribution_signed_matrix
        rt, seen = 1e-2, {}

        def scaled_off(mult_allowed):
            def f(*x, **k):
                M = orig(*x, **k)
                S = float(np.max(np.abs(M["prediction"])))
                seen["S"] = S
                d = (A_NORM_ATOL + 0.5 * rt * S) if mult_allowed is None else mult_allowed * (A_NORM_ATOL + rt * S)
                seen.setdefault("d", []).append(d)
                M["prediction"] = M["prediction"] + d
                return M
            return f
        sp_tol = make_spec(w, "anorm_tol", surv, dirs["neural_net"], votes_n, gd_n, t_s,
                           expect_checkpoint_sha256=ck["neural_net"], a_norm_rtol=rt)
        psa.per_survivor_attribution_signed_matrix = scaled_off(None)
        try:
            res, rep, *_x = cycle("anorm_tol_in", sp_tol)
            psa.per_survivor_attribution_signed_matrix = scaled_off(2.0)
            o2, d2 = refused("anorm_tol_out", sp_tol, "A-NORM")
        finally:
            psa.per_survivor_attribution_signed_matrix = orig
        in_ok = (res["status"] == "ok" and anorm_recorded_ok(rep, rt)
                 and rep["attribution"]["a_norm_max_abs_diff"] > A_NORM_ATOL)
        gate("LL8.anorm_tolerance_scales_with_scores", in_ok and o2 and len(seen.get("d", [])) == 2,
             [f"inside: {res['status']} {str(res.get('message'))[:120]} "
              f"{rep and {k: v for k, v in rep['attribution'].items() if k.startswith('a_norm_') and k != 'a_norm_scope'}}",
              f"twice allowed: {d2}", f"injected {seen.get('d')} at max|score| {seen.get('S')}"])
    if want("LL8.anorm_refuses_mismatched_forward"):
        orig = psa.per_survivor_attribution_signed_matrix
        calls = {"n": 0}

        def off_by(*x, **k):          # fault injection: an attribution forward that is not the ranking's forward
            calls["n"] += 1
            M = orig(*x, **k)
            M["prediction"] = M["prediction"] + 1e-3
            return M
        psa.per_survivor_attribution_signed_matrix = off_by
        try:
            o1, d1 = refused("anorm", sp_nn, "A-NORM")
        finally:
            psa.per_survivor_attribution_signed_matrix = orig
        gate("LL8.anorm_refuses_mismatched_forward", o1 and calls["n"] > 0, f"{d1}; injected calls {calls['n']}")
    if want("LL8.family_and_flag_checks"):
        det = [refused("fam_wrong", sp_nn, "!= the family", model_family="lightgbm"),
               refused("flag_malformed", sp_nn, "is not '<model_type>@<anchor>'", flag_id="neural_net"),
               refused("flag_space", sp_nn, "is not '<model_type>@<anchor>'", flag_id="neural_net@2026 x"),
               refused("flag_vs_family", sp_nn, "differs from the flag", model_family="neural_net",
                       flag_id="lightgbm@2026-11-26T00:00:00+00:00"),
               refused("flag_family_wrong", sp_nn, "!= the family", flag_id="xgboost@2026-11-26T00:00:00+00:00")]
        res, rep, prio, rrec, od = cycle("flag_ok", sp_nn, flag_id="neural_net@2026-11-26T01:02:03+00:00")
        good = (res["status"] == "ok" and res["model_family"] == "neural_net"
                and rrec["run_for"] == {"model_family": "neural_net", "flag_id": "neural_net@2026-11-26T01:02:03+00:00"})
        gate("LL8.family_and_flag_checks", all(o for o, _ in det) and good,
             [d for _, d in det] + [f"good flag -> {res['status']} family {res['model_family']}"])

    # ---- Q5: result file distinct, never overwritten, failures returned (in-process) ----------------
    if want("LL8.result_file_distinct_and_never_overwritten"):
        det = []
        for i, nm in enumerate(["attribution_cycle.json", "extraction_priorities.json", "signed_attribution.npz",
                                "run_record.json", "_env/x.json", "_env", ""]):
            od = w / f"q5_alias_{i}"
            rfp = od / nm if nm else od
            try:
                r = ac.run_attribution_cycle(str(sp_main), str(od), result_file=str(rfp))
                det.append((r["status"] == "refused" and r["exit_code"] == 2 and r["result_written"] is None
                            and not od.exists(), f"alias {nm or '<out-dir>'}: {r['status']} {str(r['message'])[:70]}"))
            except Exception as e:
                det.append((False, f"alias {nm}: raised {e!r}"))
        par = w / "q5_parent"
        r = ac.run_attribution_cycle(str(sp_nopin), str(par / "sub"), result_file=str(par))
        det.append((r["status"] == "refused" and not par.exists(), f"parent of out-dir: {r['status']}"))
        rfp, od = w / "q5_race.result.json", w / "q5_race"
        orig_cycle = ac._cycle

        def racing(*x, **k):          # another writer creates the result path while the cycle runs
            rfp.write_text("another writer\n")
            return orig_cycle(*x, **k)
        ac._cycle = racing
        try:
            r = ac.run_attribution_cycle(str(sp_nopin), str(od), result_file=str(rfp))
            det.append((r["status"] == "error" and r["exit_code"] == 3 and r["result_written"] is None
                        and rfp.read_text() == "another writer\n"
                        and r.get("before_publication", {}).get("status") == "refused",
                        f"race: {r['status']} kept file: {rfp.read_text()[:20]!r}; before {r.get('before_publication')}"))
        except Exception as e:
            det.append((False, f"race: raised {e!r}"))
        finally:
            ac._cycle = orig_cycle
        plain = w / "q5_plainfile"
        plain.write_text("x\n")
        try:
            r = ac.run_attribution_cycle(str(sp_nopin), str(w / "q5_pubfail"), result_file=str(plain / "r.json"))
            bp = r.get("before_publication", {})
            det.append((r["status"] == "error" and r["exit_code"] == 3 and r["result_written"] is None
                        and bp.get("status") == "refused" and "expect_checkpoint_sha256" in str(bp.get("message")),
                        f"publication failure: {r['status']} {str(r['message'])[:70]}; original kept: {bp.get('status')}"))
        except Exception as e:
            det.append((False, f"publication failure: raised {e!r}"))
        try:
            r = ac.run_attribution_cycle(str(sp_nopin), str(w / "q5_bad\0dir"))
            det.append((r["status"] == "error" and r["exit_code"] == 3 and r["result_written"] is None,
                        f"path resolution failure: {r['status']} {r['error_class']}"))
        except Exception as e:
            det.append((False, f"path resolution failure: raised {e!r}"))
        gate("LL8.result_file_distinct_and_never_overwritten", all(o for o, _ in det), [d for _, d in det])

    # ---- Q7: caller state, in a fresh child process -------------------------------------------------
    if want("LL8.caller_state_restored"):
        probe = w / "caller_probe.py"
        probe.write_text(CALLER_PROBE)
        pw = w / "probe"
        pw.mkdir()
        p = subprocess.run([sys.executable, "-B", str(probe), str(ac_file), str(REPO), str(sp_nn), str(sp_nopin), str(pw)],
                           capture_output=True, text=True, cwd=str(pw), env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        (w / "caller_probe.log").write_text(p.stdout + p.stderr)
        line = [x for x in p.stdout.splitlines() if x.startswith("PROBE ")]
        pr = json.loads(line[-1][6:]) if line else {}
        c = pr.get("calls", {})
        ok = (p.returncode == 0 and pr.get("import") == {}
              and c.get("ok", {}).get("status") == "ok" and c.get("refused", {}).get("status") == "refused"
              and c.get("setup_failure", {}).get("status") == "error"
              and all(c.get(k, {}).get("changed") == {} for k in ("ok", "refused", "setup_failure"))
              and pr.get("cuda_seen_in_cycle") == ["", ""])
        gate("LL8.caller_state_restored", ok,
             f"rc {p.returncode}; changed by import: {pr.get('import')}; "
             + "; ".join(f"{k}: {v.get('status')} changed {v.get('changed')}" for k, v in c.items())
             + f"; CUDA_VISIBLE_DEVICES inside the cycle: {pr.get('cuda_seen_in_cycle')}")

    # ---- CLI: exit status and result file; writes only under --out-dir ----------------------
    if not mutant_run and want("LL8.cli_exit_codes_and_result_file"):
        def cli(tag, extra, spec_path):
            od = w / f"cli_{tag}"
            rf = w / f"cli_{tag}.result.json"
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
            for k in ll6.ENV_ROOTS:
                env.pop(k, None)              # the script must set its own roots under --out-dir
            s0 = ll6.boundary_snapshot(od)
            p = subprocess.run([sys.executable, "-B", str(REPO / AC_REL), "--spec", str(spec_path), "--out-dir", str(od),
                                "--result-file", str(rf)] + extra, capture_output=True, text=True, env=env, cwd=str(w))
            s1 = ll6.boundary_snapshot(od)
            ch = [c for c in ll6.boundary_diff(s0, s1) if str(rf.relative_to(REPO)) not in c]
            r = json.load(open(rf)) if rf.exists() else {}
            return p.returncode, r, ch, od
        pol = {p: sha(REPO / p) for p in ["watcher_policies.json"] + sorted(str(x.relative_to(REPO)) for x in
                                                                           (REPO / "agent_manifests").glob("*.json"))}
        RUN_INFO["settings_sha_before_cli"] = pol          # taken BEFORE any child starts (Beta X-LL7-LL8 R1 Q6)
        sp_err = make_spec(w, "err", surv_p, dir_p, votes_p, gd_p, t_s, expect_checkpoint_sha256=ck["planted"], blocks=None)
        rc0, r0, ch0, od0 = cli("ok", ["--model-family", "lightgbm"], sp_main)
        outs0 = bool(r0.get("outputs")) and all(sha(od0 / n) == h for n, h in r0["outputs"].items())
        rc2, r2, ch2, _ = cli("refused", ["--model-family", "catboost"], sp_main)
        rc3, r3, ch3, _ = cli("error", [], sp_err)
        ok = (rc0 == 0 and r0.get("status") == "ok" and r0.get("exit_code") == 0
              and rc2 == 2 and r2.get("status") == "refused" and r2.get("error_class") == "CycleError"
              and rc3 == 3 and r3.get("status") == "error" and r3.get("error_class") == "KeyError"
              and "traceback_tail" in r3 and (od0 / "_env" / "mpl").is_dir() and outs0)
        env_c = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
        od_al = w / "cli_alias"
        pa = subprocess.run([sys.executable, "-B", str(REPO / AC_REL), "--spec", str(sp_main), "--out-dir", str(od_al),
                             "--result-file", str(od_al / "attribution_cycle.json")], capture_output=True, text=True,
                            env=env_c, cwd=str(w))
        plain = w / "cli_plainfile"
        plain.write_text("x\n")
        pf = subprocess.run([sys.executable, "-B", str(REPO / AC_REL), "--spec", str(sp_main), "--out-dir",
                             str(w / "cli_pubfail"), "--result-file", str(plain / "r.json"), "--model-family", "catboost"],
                            capture_output=True, text=True, env=env_c, cwd=str(w))
        cli_q5 = (pa.returncode == 2 and not od_al.exists() and "Traceback" not in pa.stderr
                  and pf.returncode == 3 and "result not published" in pf.stderr and "Traceback" not in pf.stderr
                  and "the run itself was refused" in pf.stderr and plain.read_text() == "x\n")
        gate("LL8.cli_exit_codes_and_result_file", ok and cli_q5,
             f"ok rc {rc0} {r0.get('status')} outputs verified after publication {outs0}; refused rc {rc2} "
             f"{r2.get('status')} {str(r2.get('message'))[:60]}; error rc {rc3} {r3.get('status')} {r3.get('error_class')}; "
             f"alias rc {pa.returncode}; publication failure rc {pf.returncode} stderr {pf.stderr.strip()[-160:]!r}")
        gate("LL8.writes_only_under_out_dir", not ch0 and not ch2 and not ch3,
             f"outside each --out-dir (git status, data/, /tmp/matplotlib-*, user matplotlib dirs): ok {ch0[:3]}, "
             f"refused {ch2[:3]}, error {ch3[:3]}; MPLCONFIGDIR under --out-dir: {(od0 / '_env' / 'mpl').is_dir()}")
        pol2 = {p: sha(REPO / p) for p in pol}
        gate("LL8.settings_files_unchanged", pol == pol2 and len(pol) > 1,
             f"{len(pol)} files (watcher_policies.json, agent_manifests/*.json) sha256 identical after the CLI runs")
    if not mutant_run and want("LL8.nothing_imports_attribution_cycle"):
        g = subprocess.run(["git", "-C", str(REPO), "grep", "-lE", r"(import|from)\s+.*attribution_cycle|attribution_cycle\.py",
                            "--", "*.py"], capture_output=True, text=True).stdout.split()
        g = [x for x in g if x not in (AC_REL, "tests/test_ll8_attribution_cycle.py")]
        unt = [x for x in subprocess.run(["git", "-C", str(REPO), "ls-files", "--others", "--exclude-standard", "--", "*.py"],
                                         capture_output=True, text=True).stdout.split()
               if not x.startswith("data/") and x not in (AC_REL, "tests/test_ll8_attribution_cycle.py")
               and "attribution_cycle" in (REPO / x).read_text(errors="ignore")]
        probe = ("import sys, json; sys.path.insert(0, sys.argv[1]); import prediction_generator; "
                 "print(json.dumps([m for m in sys.modules if 'attribution_cycle' in m]))")
        p = subprocess.run([sys.executable, "-B", "-c", probe, str(REPO)], capture_output=True, text=True, cwd=str(w),
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"))
        got = (p.stdout.strip().splitlines() or ["null"])[-1]
        gate("LL8.nothing_imports_attribution_cycle", not g and not unt and got == "[]",
             f"tracked .py naming it: {g}; untracked .py (outside data/) naming it: {unt}; Step 6 import probe: {got}")

    # ---- mutants -------------------------------------------------------------------------------
    mutants = []
    if not a.no_mutants and not mutant_run:
        green = {r["gate"]: r["ok"] for r in RESULTS}
        mutants = run_mutants(out, a.jobs, Path(a.models_from).resolve() if a.models_from else out, green,
                              a.baseline_rev, boundary)
        # in-process mutant: the miss report is sha-pinned, so its cohort rule is patched after loading
        if green.get("LL8.sparse_cohort_abstains"):
            calls = {"n": 0}
            orig_lm = ac.load_miss_report

            def lm(path, want_sha):
                m = orig_lm(path, want_sha)

                def ok_always(*x, **k):
                    calls["n"] += 1
                    return True, []
                m.cohort_ok = ok_always
                return m
            ac.load_miss_report = lm
            try:
                res, rep, *_r = cycle("mut_no_abstain", sp_main)
            finally:
                ac.load_miss_report = orig_lm
            killed = abstention_credit(calls["n"], res, rep, lambda r_: g_sparse(r_)[0])
            ctl = abstention_credit(1, {"status": "refused"}, None, lambda r_: False)   # control: no report, no credit
            killed = killed and not ctl
        else:
            killed, calls = False, {"n": 0}
        mutants.append({"id": "abstention_ignored", "kind": "in-process (pinned file)", "killed": killed,
                        "targets": ["LL8.sparse_cohort_abstains"], "calls": calls["n"]})
        print(f"[{'PASS' if killed else 'FAIL'}] MUTANT LL8.abstention_ignored killed={killed} calls={calls['n']}", flush=True)
    if not mutant_run and want("LL8.no_write_gate_catches_sibling_write"):
        s0 = ll6.boundary_snapshot(boundary)
        sent = REPO / "data" / f"ll8_nowrite_control_{os.getpid()}_{time.time_ns()}"
        sent.mkdir()
        (sent / "sentinel.txt").write_text("ll8 no-write negative control\n")
        s1 = ll6.boundary_snapshot(boundary)
        shutil.rmtree(sent)
        s2 = ll6.boundary_snapshot(boundary)
        caught, cleared = ll6.boundary_diff(s0, s1), ll6.boundary_diff(s0, s2)
        gate("LL8.no_write_gate_catches_sibling_write", any(str(sent.relative_to(REPO)) in c for c in caught) and not cleared,
             f"sibling sentinel seen: {caught[:2]}; after removal clean: {not cleared}")
    RUN_INFO["fixture_seconds"] = round(t_fix, 1)
    return finish(out, t0, mutants, guard_before, boundary, a)


def finish(out, t0, mutants, guard_before, boundary, a):
    ch = ll6.boundary_diff(guard_before, ll6.boundary_snapshot(boundary))
    gate("LL8.no_write_outside_out", not ch, f"changes outside {boundary}: {ch[:5]}")
    n_ok = sum(r["ok"] for r in RESULTS)
    k_ok = sum(m["killed"] for m in mutants)
    summ = {"gates": RESULTS, "mutants": mutants, "gates_pass": n_ok, "gates_total": len(RESULTS),
            "mutants_killed": k_ok, "mutants_total": len(mutants), "wall_s": round(time.time() - t0, 1),
            "rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
            "rss_children_mb": round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024),
            "run_info": RUN_INFO}
    full = not (a.no_mutants or a.ac_file or a.only)
    all_ok = n_ok == len(RESULTS) and k_ok == len(mutants) and (not full or len(mutants) == len(MUTANTS) + 1)
    (out / "selftest_summary.json").write_text(json.dumps(summ, indent=1, default=str))
    print(f"\nLL-8 self-test: {n_ok}/{len(RESULTS)} gates, {k_ok}/{len(mutants)} mutants killed, "
          f"{summ['wall_s']} s, RSS {summ['rss_mb']} MB (largest child {summ['rss_children_mb']} MB) -> "
          f"{'PASS' if all_ok else 'FAIL'}", flush=True)
    if all_ok and a.clean_on_pass and not a.ac_file:
        for d in sorted(out.iterdir()):
            if d.is_dir():
                shutil.rmtree(d)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
