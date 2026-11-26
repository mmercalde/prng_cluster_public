#!/usr/bin/env python3
"""
tests/test_ch14_diag_land_defaults.py - ch14-diag landing gate (2026-11-25, TASK_CH14_DIAG_LAND_PREP).

Proves, on the REAL Step 5 CLI, that at the defaults this landing changes nothing:
  * Step 5 run with `--enable-diagnostics` (no level flag) and with `--diagnostics-level legacy`
    writes the same models, sidecar and diagnostics reports as the code at BASE_REV;
  * the health check, at grading_rules legacy (absent key) and at an explicit `legacy`, gives the same
    result on those outputs as BASE_REV's health check;
  * non-vacuity: `--diagnostics-level full` changes the reports (training outputs stay equal), and
    grading_rules `fixed` changes the health result, on the same outputs.

BASE_REV is main at landing, by full sha (never HEAD, never a merge base). The base tree holds the
seven code files of this landing as they are at BASE_REV; the new tree holds this worktree's copies.
Every other top-level entry is a symlink into this worktree in both trees, so only those seven differ.
Both trees, every run directory, the study databases and all reports are written under --out (a NEW
directory under <worktree>/data/). Nothing outside --out is written; the suite checks the worktree's
`git status --porcelain --ignored` before and after.

Step 5 is run through main's scripts/diagnostics/e2e_step5_shim.py with a fixed --split-seed, because
the module's 80/20 split is unseeded (docs/CHAPTER_6_AUDIT.md:140; TODO-E2E-21); the shim changes nothing
else. A control (BASE_REV run twice) shows which outputs are deterministic; a field that differs in the
control is reported and excluded from that comparison, never silently.

The neural net is not reproducible run to run on CPU, even at BASE_REV (no torch seed on the Step 5
path), so its outputs are compared by key structure, and the command lines of the children Step 5 starts
(the NN trial worker, recorded by the shim) are compared exactly. Its trainer's values are compared to
BASE_REV in-process with torch seeded (tests/test_ch14_diag_wave1.py::test_nn_legacy_report_identical_to_base).

Added after Beta X-CH14-DIAG R1 (lead GO 2026-11-25 20:02), each with a mutant the R1 gates missed:
  (1) the final NN call's --params equal the run's own study.best_trial.params        (S05: worst trial passed)
  (2) what the real worker main() hands train_neural_net (params, kwargs, array sha256s), base against new
  (3) weights and predictions from that worker run, seeded by the test driver only    (S06: lr changed in the worker)
  (4) --part compare: --compare-models --enable-diagnostics, base against new          (S07: compare child flag)

Mutants (applied to a copy of the new tree; each must make its gate FAIL, with the mutated line shown
to have run): S01 legacy treated as full in Step 5; S02 a training change in the Step 5 trainer;
S03 legacy health rules silently fixed; S04 the NN child gets an extra flag at legacy; S05-S07 above.
No test here runs `git commit` (owner ruling 2026-11-25 19:26).

CPU only: CUDA_VISIBLE_DEVICES="", S88_FORCE_LGBM_CPU=1, --device cpu.
    taskset -c 3-5 nice -n 10 env PYTHONDONTWRITEBYTECODE=1 python3 -u -B tests/test_ch14_diag_land_defaults.py \
        --out data/ch14_land/defaults_<ts>
"""
import argparse
import hashlib
import json
import os
import random
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parent.parent
BASE_REV = "7f4505aaccebdd284393eb3565e75d09d1c0f3b4"
CODE_FILES = ("meta_prediction_optimizer_anti_overfit.py", "train_single_trial.py", "training_diagnostics.py",
              "training_health_check.py", "diagnostics_llm_analyzer.py", "parameter_advisor.py",
              "agents/contexts/advisor_bundle.py")
# (R1's SKIP_TOP list is replaced by OUTPUT_TOP and tracked-only linking in build_tree; Beta R1 Q4)
FAMILIES = ("lightgbm", "xgboost", "catboost", "neural_net")
SPLIT_SEED = 1125
HIT = "__import__('ch14_mutant_mark').hit()"
MARK_MODULE = 'import os\ndef hit():\n    open(os.environ["CH14_MUTANT_MARK"], "a").write("hit\\n")\n    return True\n'

RESULTS = []


def gate(name, ok, detail=""):
    RESULTS.append({"gate": name, "ok": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    return ok


def sha_b(b):
    return hashlib.sha256(b).hexdigest()


def git(*a):
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, check=True).stdout


# ---------------------------------------------------------------------------- trees
# Output places a run writes to. They are never linked into a test tree, whether tracked or not.
OUTPUT_TOP = {".git", "data", "models", "diagnostics_outputs", "optuna_studies", "agents", "__pycache__",
              "logs", ".pytest_cache", "catboost_info"}


def _tracked(sub: str = "") -> set:
    """Names git tracks at HEAD directly under `sub` of this worktree (top level when empty)."""
    args = ["git", "-C", str(REPO), "ls-tree", "--name-only", "HEAD"] + ([f"{sub}/"] if sub else [])
    names = subprocess.run(args, capture_output=True, text=True, check=True).stdout.split()
    return {n.rsplit("/", 1)[-1] for n in names}


def build_tree(dst: Path, sources: dict, link_policy: str = "tracked", src_root: Path = None):
    """A test code tree: the seven code files from `sources`; everything else a symlink into this worktree.

    link_policy 'tracked' (the default; Beta X-CH14-DIAG R1 Q4 fix): only entries git TRACKS at HEAD are
    linked. An ignored or untracked entry (for example a pre-existing catboost_info/ from an earlier run)
    is never linked, so CatBoost's train dir, which it creates in its process's cwd (= this tree), is always a
    real local directory under --out. Output places (OUTPUT_TOP) are never linked; models/, agents/ and
    agents/contexts/ are real directories of tracked links; tracked top-level .jsonl logs are COPIED, so an
    append lands in the copy.
    link_policy 'legacy' is the R1 behaviour (everything except a fixed skip list), kept only for the
    negative control, which must show that it writes through a planted catboost_info/."""
    src_root = Path(src_root) if src_root is not None else REPO   # the negative control passes a disposable one
    dst.mkdir(parents=True)
    legacy = link_policy == "legacy"
    top = None if legacy else _tracked("")
    skip = {".git", "data", "models", "diagnostics_outputs", "optuna_studies", "agents", "__pycache__",
            "logs", ".pytest_cache"} if legacy else OUTPUT_TOP
    for e in src_root.iterdir():
        if e.name in skip or e.name in CODE_FILES:
            continue
        if not legacy and e.name not in top:
            continue
        if not legacy and e.is_file() and e.suffix == ".jsonl":
            shutil.copyfile(e, dst / e.name)
            continue
        os.symlink(e, dst / e.name)
    for sub in ("models", "agents"):
        (dst / sub).mkdir()
        tr = None if legacy else _tracked(sub)
        for e in (src_root / sub).iterdir():
            if e.name in ("__pycache__",) or f"{sub}/{e.name}" in CODE_FILES:
                continue
            if not legacy and e.name not in tr:
                continue
            if sub == "models" and e.is_dir() and e.name not in ("wrappers",):
                continue   # models/reinforcement and the test_* model dirs are OUTPUT dirs: never linked
            if sub == "agents" and e.name == "contexts":
                continue
            os.symlink(e, dst / sub / e.name)
    (dst / "agents" / "contexts").mkdir()
    trc = None if legacy else _tracked("agents/contexts")
    for e in (src_root / "agents" / "contexts").iterdir():
        if e.name == "__pycache__" or f"agents/contexts/{e.name}" in CODE_FILES:
            continue
        if not legacy and e.name not in trc:
            continue
        os.symlink(e, dst / "agents" / "contexts" / e.name)
    for rel, data in sources.items():
        (dst / rel).write_bytes(data)
    (dst / "ch14_mutant_mark.py").write_text(MARK_MODULE)
    return dst


# ---------------------------------------------------------------- Beta R3 Q5: temp/cache roots and the boundary
# The pattern of tests/test_ll6_inference_bundle.py:85-140 on main (Beta X-LL6-BUNDLE R1 Q5), with sha256 added.
ENV_ROOTS = {"TMPDIR": "tmp", "XDG_CACHE_HOME": "cache", "MPLCONFIGDIR": "mpl", "CUDA_CACHE_PATH": "cuda_cache"}
USER_CACHE_DIRS = [Path.home() / ".cache" / "matplotlib", Path.home() / ".config" / "matplotlib"]


def redirect_env_roots(out: Path):
    """Temp and cache roots under --out, set before any third-party import and inherited by every child (every
    subprocess here is started with env=dict(os.environ, ...)). LightGBM's matplotlib import otherwise makes
    /tmp/matplotlib-* when the user cache is not writable (Beta R3 Q5, :99)."""
    import tempfile
    for k, d in ENV_ROOTS.items():
        (out / "_env" / d).mkdir(parents=True, exist_ok=True)
        os.environ[k] = str(out / "_env" / d)
    tempfile.tempdir = None
    return {k: os.environ[k] for k in ENV_ROOTS}


def _file_sig(q: Path):
    st = q.lstat()
    if q.is_symlink():
        return ("link", os.readlink(q))
    return (st.st_size, st.st_mtime_ns, sha_b(q.read_bytes()))


def boundary_snapshot(out: Path):
    """Everything the suite must not write, outside the selected --out subtree:
    (a) every file under data/ with only --out pruned: (size, mtime_ns, sha256); symlinks by target, never followed;
    (b) every file of the worktree outside data/ and .git: the same signature;
    (c) git status of the worktree outside data/ (tracked, untracked, ignored);
    (d) new /tmp/matplotlib-* entries, and the user's matplotlib cache and config directories."""
    o = str(out)
    data = {}
    for root, dirs, files in os.walk(REPO / "data", followlinks=False):
        dirs[:] = [d for d in dirs if str(Path(root) / d) != o and not str(Path(root) / d).startswith(o + "/")]
        for n in files + [d for d in dirs if (Path(root) / d).is_symlink()]:
            q = Path(root) / n
            try:
                data[str(q.relative_to(REPO))] = _file_sig(q)
            except (FileNotFoundError, PermissionError):
                pass
    tree = {}
    for root, dirs, files in os.walk(REPO, followlinks=False):
        rel = Path(root).relative_to(REPO)
        if rel == Path("."):
            dirs[:] = [d for d in dirs if d not in ("data", ".git")]
        for n in files:
            q = Path(root) / n
            if rel == Path(".") and n == ".git":
                continue
            try:
                tree[str(q.relative_to(REPO))] = _file_sig(q)
            except (FileNotFoundError, PermissionError):
                pass
    st = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--ignored", "--untracked-files=all",
                         "--", ".", ":(exclude)data"], capture_output=True, text=True).stdout.splitlines()
    tmp = sorted(e.name for e in Path("/tmp").iterdir() if e.name.startswith("matplotlib-"))
    caches = {}
    for c in USER_CACHE_DIRS:
        if c.exists():
            for e in c.iterdir():
                s_ = e.lstat()
                caches[str(e)] = (s_.st_size, s_.st_mtime_ns)
    return {"data": data, "tree": tree, "git": sorted(st), "tmp_matplotlib": tmp, "user_caches": caches}


def boundary_diff(a, b):
    ch = []
    for part in ("data", "tree"):
        for k in sorted(set(a[part]) | set(b[part])):
            if a[part].get(k) != b[part].get(k):
                ch.append(f"{part}: {k} {'new' if k not in a[part] else ('gone' if k not in b[part] else 'changed')}")
    ch += [f"git: {x}" for x in sorted(set(b["git"]) ^ set(a["git"]))]
    ch += [f"/tmp: {x}" for x in sorted(set(b["tmp_matplotlib"]) - set(a["tmp_matplotlib"]))]
    for k in sorted(set(a["user_caches"]) | set(b["user_caches"])):
        if a["user_caches"].get(k) != b["user_caches"].get(k):
            ch.append(f"user cache: {k}")
    return ch


def boundary_negative_control(out: Path):
    """Overwrite an EXISTING sibling file under data/ outside --out (same size, mtime restored, so only the
    content differs) and require the boundary gate to turn red; restore bytes and mtime and require it green."""
    cands = []
    for root, dirs, files in os.walk(REPO / "data", followlinks=False):
        o = str(out)
        dirs[:] = sorted(d for d in dirs if str(Path(root) / d) != o and not str(Path(root) / d).startswith(o + "/"))
        for n in sorted(files):
            q = Path(root) / n
            if not q.is_symlink() and q.is_file() and q.stat().st_size > 0 \
                    and time.time() - q.stat().st_mtime > 120:      # never a live log
                cands.append(q)
        if cands:
            break
    if not cands:
        gate("BOUNDARY.negative_control_has_a_sibling_file", False, "no existing file under data/ outside --out")
        return
    t = cands[0]
    orig, st = t.read_bytes(), t.stat()
    a = boundary_snapshot(out)
    try:
        t.write_bytes(bytes([orig[0] ^ 0x01]) + orig[1:])
        os.utime(t, ns=(st.st_atime_ns, st.st_mtime_ns))
        red = boundary_diff(a, boundary_snapshot(out))
    finally:
        t.write_bytes(orig)
        os.utime(t, ns=(st.st_atime_ns, st.st_mtime_ns))
    green = boundary_diff(a, boundary_snapshot(out))
    gate("BOUNDARY.negative_control: a same-size overwrite of an existing sibling under data/ (mtime restored) "
         "turns the gate RED", bool(red), f"target {t.relative_to(REPO)}; detected {red[:3]}")
    gate("BOUNDARY.negative_control: restored, the gate is GREEN again", not green and t.read_bytes() == orig,
         f"changes after restore {green[:3]}")


# ---------------------------------------------------------------- 2. isolation of existing directories
MAIN_CHECKOUT = Path.home() / "distributed_prng_analysis"
ISOLATION_NAMES = ("catboost_info",)


def isolation_roots(extra=()):
    """Every existing catboost_info/ (a) anywhere in this worktree outside data/ and .git, (b) directly in
    each parent of the worktree up to the home directory, and (c) in the main checkout (depth 4, data/
    and .git pruned). Symlinks are never followed."""
    roots = set(Path(x) for x in extra)
    def walk(base: Path, maxdepth: int):
        if not base.is_dir():
            return
        for dp, dn, fn in os.walk(base, followlinks=False):
            rel = Path(dp).relative_to(base)
            dn[:] = [d for d in dn if d not in (".git", "data", "__pycache__") and not (Path(dp) / d).is_symlink()]
            if len(rel.parts) >= maxdepth:
                dn[:] = []
            for name in ISOLATION_NAMES:
                if name in dn or (Path(dp) / name).is_dir():
                    roots.add(Path(dp) / name)
    walk(REPO, 6)
    p = REPO.parent
    while True:
        for name in ISOLATION_NAMES:
            if (p / name).is_dir():
                roots.add(p / name)
        if p == Path.home() or p == p.parent:
            break
        p = p.parent
    walk(MAIN_CHECKOUT, 4)
    return sorted(roots)


def isolation_snapshot(roots):
    """{path: (size, mtime_ns, sha256)} for every regular file under each root, plus {root: 'DIR'|'ABSENT'}."""
    snap = {}
    for r in roots:
        r = Path(r)
        snap[str(r)] = "DIR" if r.is_dir() else "ABSENT"
        if not r.is_dir():
            continue
        for dp, dn, fn in os.walk(r, followlinks=False):
            for f in fn:
                q = Path(dp) / f
                if q.is_symlink() or not q.is_file():
                    continue
                st = q.stat()
                snap[str(q)] = (st.st_size, st.st_mtime_ns, sha_b(q.read_bytes()))
    return snap


def mutate(tree: Path, rel: str, old: str, new: str):
    p = tree / rel
    s = p.read_text()
    if s.count(old) != 1:
        raise SystemExit(f"mutant text not found exactly once in {rel}: {old[:80]!r}")
    p.write_text(s.replace(old, new))


# ---------------------------------------------------------------------------- fixture
def make_fixture(out: Path):
    names = [l for l in (REPO / "tests/fixtures/runH_step3_feature_names.txt").read_text().splitlines()
             if l and not l.startswith("#")]
    r = random.Random(20261125)
    rows = []
    for i in range(240):
        feats = {n: round(r.gauss(0.0, 1.0), 6) for n in names}
        feats["score"] = round(r.random(), 6)
        feats["confidence"] = round(r.random(), 6)
        rows.append({"seed": r.randrange(2 ** 32), "features": feats, "score": feats["score"],
                     "holdout_hits": float(r.randrange(0, 4)), "holdout_quality": round(r.random(), 6)})
    s5 = out / "fixture_survivors.json"
    s5.write_text(json.dumps(rows))
    lot = out / "fixture_lottery.json"
    lot.write_text(json.dumps([r.randrange(1000) for _ in range(300)]))
    return s5, lot


# ---------------------------------------------------------------------------- Step 5 runs
def run_step5(sources: dict, d: Path, s5: Path, lot: Path, family: str, extra, mark=None, mutation=None):
    """One Step 5 CLI call. d becomes a fresh code tree (build_tree), and Step 5 runs with cwd = d,
    because production runs from the repo root: the final NN subprocess is started as a relative
    'train_single_trial.py' (meta_prediction_optimizer_anti_overfit.py:2631 at 7f4505a). Every file the
    run creates is therefore a regular file under d."""
    build_tree(d, sources)
    if mutation:
        mutate(d, *mutation)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", S88_FORCE_LGBM_CPU="1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=str(d), CH14_MUTANT_MARK=str(mark or (d / "unused.mark")))
    argv = [sys.executable, "-B", str(REPO / "scripts/diagnostics/e2e_step5_shim.py"),
            "--split-seed", str(SPLIT_SEED), "--capture-dir", str(d / "shim"),
            "--script", str(d / "meta_prediction_optimizer_anti_overfit.py"), "--",
            "--survivors", str(s5), "--lottery-data", str(lot), "--model-type", family,
            "--trials", "2", "--k-folds", "2", "--holdout-draws", "60", "--device", "cpu",
            "--output-dir", str(d / "models" / "reinforcement"), "--study-name", f"ch14land_{family}",
            "--storage", f"sqlite:///{d / 'optuna.db'}", "--enable-diagnostics"] + list(extra)
    t0 = time.time()
    p = subprocess.run(argv, cwd=d, env=env, capture_output=True, text=True)
    (d / "cli.log").write_text(f"argv {argv}\nrc {p.returncode}\n--- stdout\n{p.stdout}\n--- stderr\n{p.stderr}")
    return {"rc": p.returncode, "seconds": round(time.time() - t0, 1), "dir": d}


VOLATILE_KEYS = {"generated_at", "timestamp", "created_at", "saved_at", "start_time", "end_time", "started_at",
                 "completed_at", "duration", "duration_seconds", "elapsed", "elapsed_seconds", "training_time",
                 "train_time"}
import re as _re
_TIME_ID = _re.compile(r"^(step5|S88|s5)[_-]\d{8}[_T-]?\d{6}.*$")   # ids made from the wall clock


def scrub(obj, roots):
    """Drop wall-clock keys; replace run-specific absolute paths by placeholders."""
    if isinstance(obj, dict):
        return {k: scrub(v, roots) for k, v in obj.items() if k not in VOLATILE_KEYS}
    if isinstance(obj, list):
        return [scrub(v, roots) for v in obj]
    if isinstance(obj, str):
        for src, ph in roots:
            obj = obj.replace(src, ph)
        return "<TIME_ID>" if _TIME_ID.match(obj) else obj
    return obj


def study_digest(db: Path):
    """An Optuna sqlite study compared by content: every trial's number, state, value and params
    (the file bytes hold wall-clock times). Opened read-only and immutable (no -wal/-shm written)."""
    import sqlite3
    con = sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)
    try:
        rows = con.execute("SELECT t.number, t.state, v.value FROM trials t LEFT JOIN trial_values v "
                           "ON v.trial_id = t.trial_id ORDER BY t.number").fetchall()
        params = con.execute("SELECT t.number, p.param_name, p.param_value FROM trial_params p JOIN trials t "
                             "ON t.trial_id = p.trial_id ORDER BY t.number, p.param_name").fetchall()
    finally:
        con.close()
    return sha_b(json.dumps({"trials": rows, "params": params}).encode())


def key_paths(obj, prefix=""):
    """The set of JSON key paths (lists collapsed to []): the report's structure, not its values."""
    out = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.add(f"{prefix}/{k}")
            out |= key_paths(v, f"{prefix}/{k}")
    elif isinstance(obj, list):
        for v in obj:
            out |= key_paths(v, f"{prefix}[]")
    return out


# The NN's winning trial is chosen at random run to run, even at BASE_REV: each run proposes the same
# trials (TPE seed 42) but their scores differ, so the winner, and with it the final model's architecture,
# differs (measured: two BASE_REV runs chose a 3-layer and a 4-layer winner). These paths describe WHICH
# trial won, and are masked in the structure comparison; the winner itself is checked in child_argv_match.
WINNER_PATHS = ("/hyperparameters",)


def structure(run, rels):
    """{rel: sha of the sorted key-path set} for the JSON files in rels (non-JSON: file present)."""
    d = run["dir"]
    res = {}
    for rel in rels:
        p = d / rel
        if not p.is_file():
            continue
        if p.suffix == ".json":
            try:
                kp = key_paths(scrub(json.loads(p.read_text()), []))
                kp = {k for k in kp if not any(k == w or k.startswith(w + "/") for w in WINNER_PATHS)}
                res[rel] = sha_b("\n".join(sorted(kp)).encode())
                continue
            except ValueError:
                pass
        res[rel] = "present"
    return res


def child_calls(run):
    """The child processes Step 5 started (argv as main's shim records them in calls.jsonl), with the
    run dir, the per-export mkdtemp name and the NPZ content-hash suffix normalised."""
    f = run["dir"] / "shim" / "calls.jsonl"
    out = []
    if not f.is_file():
        return out
    for line in f.read_text().splitlines():
        argv = json.loads(line).get("argv") or []
        norm = []
        for x in (argv if isinstance(argv, list) else [argv]):
            x = str(x).replace(str(run["dir"]), "<RUN>")
            x = _re.sub(r"(step5_nn_)[A-Za-z0-9_]{8}", r"\1<TMP>", x)
            x = _re.sub(r"(split_t\w+?_f\w+?_)[0-9a-f]{12}\.npz", r"\1<H>.npz", x)
            x = norm_s88(x)
            norm.append(x)
        out.append(norm)
    return out


def _final_params_as_trial(pj):
    """Step 5's final-model --params (n_layers, layer_i, patience) in the trial calls' form."""
    p = json.loads(pj)
    if "n_layers" not in p:
        return p
    return {"hidden_layers": [p[f"layer_{i}"] for i in range(int(p["n_layers"]))], "dropout": p.get("dropout"),
            "learning_rate": p.get("learning_rate"), "batch_size": p.get("batch_size"), "epochs": p.get("epochs"),
            "early_stopping_patience": p.get("patience")}


def child_argv_match(cb, cn):
    """(ok, detail). Trial calls (all but the last) must be identical. The last call (the final model) must
    be identical except its --params value, which must be the params of one of this run's trial calls
    (the winner is one of the same, deterministic proposals; which one wins is not deterministic)."""
    if len(cb) != len(cn):
        return False, f"{len(cn)} calls vs base {len(cb)}"
    if not cb:
        return True, "0 child calls"
    if cb[:-1] != cn[:-1]:
        i = next(i for i, (x, y) in enumerate(zip(cb, cn)) if x != y)
        return False, f"trial call {i} differs"
    def strip(a):
        a = list(a)
        if "--params" in a:
            j = a.index("--params")
            return a[:j] + a[j + 2:], a[j + 1]
        return a, None
    fb, pb = strip(cb[-1])
    fn, pn = strip(cn[-1])
    if fb != fn:
        return False, f"final call differs outside --params: {sorted(set(fn) ^ set(fb))[:6]}"
    if pn is None:
        return pb is None, "no --params"
    trial_params = []
    for c in cn[:-1]:
        if "--params" in c:
            trial_params.append(json.loads(c[c.index("--params") + 1]))
    ok = _final_params_as_trial(pn) in trial_params
    return ok, (f"{len(cn) - 1} trial calls identical; final call identical outside --params; its winner "
                f"{'is' if ok else 'is NOT'} one of the {len(trial_params)} trial proposals")


def best_trial_params(run):
    """study.best_trial.params of the run's own NN study (optuna_studies/*neural_net*.db), read from a COPY
    so the study file is never opened for writing."""
    import optuna
    dbs = sorted((run["dir"] / "optuna_studies").glob("*neural_net*.db"))
    if len(dbs) != 1:
        return None
    cp = run["dir"].parent / f"{run['dir'].name}__study_copy__hc.db"
    shutil.copyfile(dbs[0], cp)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st = f"sqlite:///{cp}"
    names = optuna.get_all_study_names(storage=st)
    if len(names) != 1:
        return None
    return dict(optuna.load_study(study_name=names[0], storage=st).best_trial.params)


def final_matches_best_trial(run):
    """(ok, detail): the final NN child's --params equal the run's own study.best_trial.params."""
    cs = child_calls(run)
    if not cs or "--params" not in cs[-1]:
        return False, "no final call with --params"
    fin = json.loads(cs[-1][cs[-1].index("--params") + 1])
    best = best_trial_params(run)
    return fin == best, (f"final {fin.get('n_layers')} layers lr {fin.get('learning_rate')}; best trial "
                         f"{(best or {}).get('n_layers')} layers lr {(best or {}).get('learning_rate')}")


WORKER_DRIVER = r'''
"""Test-only: run the real train_single_trial.main() with a given argv, torch/numpy/random seeded HERE
(never in the worker), and record exactly what main() hands to train_neural_net."""
import hashlib, json, os, random, sys
tree, argv_json, record = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, tree)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, torch
torch.manual_seed(20261125); np.random.seed(20261125); random.seed(20261125)
torch.set_num_threads(1)
import train_single_trial as t
orig = t.train_neural_net
rec = {}
def arr(a):
    a = np.ascontiguousarray(a)
    return {"sha256": hashlib.sha256(a.tobytes()).hexdigest(), "shape": list(a.shape), "dtype": str(a.dtype)}
def wrapped(X_train, y_train, X_val, y_val, params, save_path=None, enable_diagnostics=False, **kw):
    rec.update({"arrays": {n: arr(v) for n, v in (("X_train", X_train), ("y_train", y_train),
                                                  ("X_val", X_val), ("y_val", y_val))},
                "params": params, "save_path_name": os.path.basename(save_path or ""),
                "enable_diagnostics": enable_diagnostics,
                "kwargs": {k: (v if k != "diagnostics_options" else (None if v is None else sorted(v)))
                           for k, v in kw.items()}})
    return orig(X_train, y_train, X_val, y_val, params, save_path, enable_diagnostics, **kw)
t.train_neural_net = wrapped
argv = json.loads(argv_json)
sys.argv = ["train_single_trial.py"] + argv
rc = t.main()
out = {"rc": rc, "boundary": rec}
ck = [os.path.join(dp, f) for dp, _, fs in os.walk("_worker_models") for f in fs if f.endswith(".pth")]
out["checkpoints"] = len(ck)
if len(ck) == 1:
    c = torch.load(ck[0], map_location="cpu", weights_only=False)
    out["weights"] = {k: hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()
                      for k, v in c["state_dict"].items()}
    from models.wrappers.neural_net_wrapper import SurvivorQualityNet
    m = SurvivorQualityNet(input_size=c["feature_count"], hidden_layers=c["hidden_layers"],
                           dropout=c["dropout"], use_leaky_relu=c["use_leaky_relu"])
    m.load_state_dict(c["state_dict"]); m.eval()
    z = np.load(argv[argv.index("--data-path") + 1])
    X = z["X_val"].astype(np.float32)
    if c.get("scaler_mean") is not None:
        X = ((X - c["scaler_mean"]) / c["scaler_scale"]).astype(np.float32)
    with torch.no_grad():
        p = m(torch.from_numpy(X)).numpy().ravel().astype(np.float64)
    p = p * float(c.get("y_std") or 1.0) + float(c.get("y_mean") or 0.0)
    out["predictions_sha256"] = hashlib.sha256(p.tobytes()).hexdigest()
    out["predictions_head"] = [float(x) for x in p[:3]]
json.dump(out, open(record, "w"), indent=1, sort_keys=True, default=str)
'''


def worker_npz(out: Path):
    """A fixed NPZ shaped like Step 5's private handoff (X_train, y_train, X_val, y_val), 89 columns."""
    import numpy as np
    p = out / "worker_boundary_split.npz"
    if not p.exists():
        rng = np.random.default_rng(1125)
        np.savez(p, X_train=rng.normal(size=(160, 89)).astype(np.float32), y_train=rng.random(160).astype(np.float32),
                 X_val=rng.normal(size=(40, 89)).astype(np.float32), y_val=rng.random(40).astype(np.float32))
    return p


def worker_argv_from(run, npz: Path, extra=()):
    """The final NN call Step 5 made in `run`, with its data path and model dir pointed at fixed local ones
    (relative '_worker_models', inside the worker's own tree)."""
    a = list(child_calls(run)[-1][2:])          # drop the interpreter and the script name
    a[a.index("--data-path") + 1] = str(npz)
    if "--model-output-dir" in a:
        a[a.index("--model-output-dir") + 1] = "_worker_models"
    return a + list(extra)


def run_worker(sources: dict, d: Path, argv, mutation=None, mark=None):
    """The real worker CLI (train_single_trial.main) in a fresh tree d, through WORKER_DRIVER."""
    build_tree(d, sources)
    if mutation:
        mutate(d, *mutation)
    (d / "_worker_driver.py").write_text(WORKER_DRIVER)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(d),
               CH14_MUTANT_MARK=str(mark or (d / "unused.mark")))
    rec = d / "_boundary_record.json"
    p = subprocess.run([sys.executable, "-B", str(d / "_worker_driver.py"), str(d), json.dumps(argv), str(rec)],
                       cwd=d, env=env, capture_output=True, text=True)
    (d / "worker.log").write_text(p.stdout + p.stderr)
    if p.returncode != 0 or not rec.is_file():
        return {"_error": (p.stderr or p.stdout)[-800:]}
    return json.loads(rec.read_text())


def norm_s88(x: str) -> str:
    return _re.sub(r"S88_\d{8}_\d{6}", "<S88>", x)


def compare_digest(run):
    """The compare-models summary reduced to what must not change: per family, its return code and which
    artifacts were archived (scores and wall-clock ids are dropped)."""
    d = run["dir"] / "diagnostics_outputs"
    fs = sorted(p for p in d.glob("compare_models_summary_S88_*.json"))
    if len(fs) != 1:
        return {"_summaries": len(fs)}
    j = json.loads(fs[0].read_text())
    return {m: {"returncode": v.get("returncode"), "archived": v.get("archived_artifacts")}
            for m, v in sorted((j.get("models") or {}).items())}


def training_part(snap):
    """The training outputs: everything except diagnostics reports and history."""
    return {k: v for k, v in snap.items() if "diagnostics" not in k and "history" not in k}


def diff_keys(a, b):
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


# ---------------------------------------------------------------------------- health check
def health(tree: Path, run_dir: Path, policies, mark=None, label="hc"):
    """tree = the code tree whose training_health_check runs; run_dir = the run whose reports are checked."""
    """Run tree's training_health_check.check_training_health() in a COPY of the run's reports,
    so the check's own registry/archive writes stay out of the compared outputs."""
    w = run_dir.parent / f"{run_dir.name}__{label}__hc"
    shutil.rmtree(w, ignore_errors=True)
    w.mkdir()
    src = run_dir / "diagnostics_outputs"
    if src.exists():
        shutil.copytree(src, w / "diagnostics_outputs")
    (w / "watcher_policies.json").write_text(json.dumps({"training_diagnostics": policies}))
    code = ("import json,sys; sys.path.insert(0, sys.argv[1]); import training_health_check as t; "
            "r = t.check_training_health(); print(json.dumps(r, sort_keys=True, default=str))")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(tree),
               CH14_MUTANT_MARK=str(mark or (w / "unused.mark")))
    p = subprocess.run([sys.executable, "-B", "-c", code, str(tree)], cwd=w, env=env, capture_output=True, text=True)
    (w / "hc.log").write_text(p.stdout + p.stderr)
    if p.returncode != 0:
        return {"_error": p.stderr[-800:]}
    r = json.loads(p.stdout.strip().splitlines()[-1])
    return scrub(r, [(str(w), "<HC>")])


NOT_OUTPUTS = set(CODE_FILES) | {"ch14_mutant_mark.py", "cli.log", "unused.mark"}


def snapshot_after(run):
    """{relpath: digest} for every regular file the run created under its tree (symlinks into the
    worktree are skipped and never followed). JSON is scrubbed; study databases compared by trials."""
    d = run["dir"]
    roots = [(str(d), "<RUN>")]
    snap = {}
    for dirpath, dirnames, filenames in os.walk(d, followlinks=False):
        dp = Path(dirpath)
        rel_dir = dp.relative_to(d)
        if rel_dir.parts[:1] == ("shim",) or any(x.endswith("__hc") for x in rel_dir.parts):
            dirnames[:] = []
            continue
        for fn in filenames:
            p = dp / fn
            rel = str(p.relative_to(d))
            if p.is_symlink() or rel in NOT_OUTPUTS or fn.startswith("_worker") or fn == "worker.log":
                continue
            rel = norm_s88(rel)
            if p.suffix == ".db":
                snap[rel] = study_digest(p)
                continue
            if p.suffix == ".json":
                try:
                    snap[rel] = sha_b(json.dumps(scrub(json.loads(p.read_text()), roots), sort_keys=True).encode())
                    continue
                except ValueError:
                    pass
            snap[rel] = sha_b(p.read_bytes())
    return snap


# ---------------------------------------------------------------------------- main
def boundary_norm(rec):
    """The worker-to-trainer record, with the one kwarg this landing adds (diagnostics_options) treated as
    absent when it is None: train_neural_net's new parameter defaults to None, so None == not passed."""
    b = json.loads(json.dumps(rec.get("boundary") or {}, default=str))
    kw = b.get("kwargs") or {}
    if kw.get("diagnostics_options", "x") is None:
        kw.pop("diagnostics_options")
    return b


def single_part_extras(out, runs, srcs, s5, lot, families):
    """Beta X-CH14-DIAG R1 Q2 gaps (lead GO, 2026-11-25 20:02): (1) the final NN call's params are the run's
    own study.best_trial; (2) what the worker hands the trainer; (3) seeded values through the real worker
    CLI. Each has a mutant that the R1 gates missed (shown) and the new gate catches."""
    if "neural_net" not in families:
        return
    fam = "neural_net"
    # (1) winner = best trial, on every NN run (BASE_REV's code obeys the same rule)
    for n in ("base", "base2", "new_default", "new_legacy"):
        gate(f"R2.{fam}.{n}_final_params_equal_own_best_trial", *final_matches_best_trial(runs[(fam, n)]))
    # (2)+(3) the worker boundary and seeded values, base against new, on Step 5's own final command line
    npz = worker_npz(out)
    argv = worker_argv_from(runs[(fam, "base")], npz)
    W = {}
    for name, tk, extra in (("base", "base", []), ("base2", "base", []), ("new", "new", []),
                            ("new_full", "new", ["--diagnostics-level", "full"])):
        W[name] = run_worker(srcs[tk], out / "worker" / name, argv + extra)
    ok_all = all("_error" not in w and w.get("rc") == 0 and w.get("checkpoints") == 1 for w in W.values())
    gate("R2.worker.all_runs_ok", ok_all, str({k: (w.get("rc"), w.get("checkpoints"), w.get("_error", "")[:80])
                                               for k, w in W.items()}))
    if not ok_all:
        return
    same = lambda a, b: (boundary_norm(a) == boundary_norm(b), a["weights"] == b["weights"],
                         a["predictions_sha256"] == b["predictions_sha256"])
    c = same(W["base"], W["base2"])
    gate("R2.worker.control_seeded_base_twice_identical", all(c),
         f"boundary/weights/predictions equal: {c} (the seeded worker is deterministic, so the value check is real)")
    n = same(W["base"], W["new"])
    gate("R2.worker.new_legacy_boundary_weights_predictions_equal_base", all(n),
         f"boundary/weights/predictions equal: {n}; predictions head {W['new']['predictions_head']}")
    f = same(W["base"], W["new_full"])
    gate("R2.worker.full_changes_boundary_not_values", (not f[0]) and f[1] and f[2],
         f"boundary equal {f[0]} (full passes diagnostics_options), weights {f[1]}, predictions {f[2]}")

    mm = out / "mutants_r2"
    # S05: Step 5 hands the final model the WORST trial (Beta's swapped-winner case)
    m5 = ("meta_prediction_optimizer_anti_overfit.py",
          "        self.best_config = dict(study.best_trial.params)\n",
          "        self.best_config = dict((min(study.trials, key=lambda _t: _t.value if _t.value is not None "
          "else float('inf')) if " + HIT + " else study.best_trial).params)\n")
    mark5 = mm / "S05" / "mark"
    (mm / "S05").mkdir(parents=True)
    r5 = run_step5(srcs["new"], mm / "S05" / "run", s5, lot, fam, ["--diagnostics-level", "legacy"], mark5, mutation=m5)
    old_ok = child_argv_match(child_calls(runs[(fam, "base")]), child_calls(r5))[0]
    new_ok, det5 = final_matches_best_trial(r5)
    hits5 = mark5.read_text().count("hit") if mark5.exists() else 0
    gate("MUTANT S05_final_model_gets_worst_trial killed by the new gate, missed by the R1 gate",
         hits5 > 0 and r5["rc"] == 0 and old_ok and not new_ok,
         f"marker hits {hits5}; R1 child-argv gate {'PASS (missed)' if old_ok else 'FAIL'}; "
         f"new best-trial gate {'PASS' if new_ok else 'FAIL'}: {det5}")
    # S06: the worker changes the learning rate after json.loads(args.params) (Beta's CLI-only case)
    m6 = ("train_single_trial.py",
          "        params = json.loads(args.params)\n",
          "        params = json.loads(args.params)\n        if " + HIT + ":\n"
          "            params['learning_rate'] = params.get('learning_rate', 0.001) * 1.5\n")
    mark6 = mm / "S06" / "mark"
    (mm / "S06").mkdir(parents=True)
    r6 = run_step5(srcs["new"], mm / "S06" / "run", s5, lot, fam, ["--diagnostics-level", "legacy"], mark6, mutation=m6)
    S_b, S_b2 = snapshot_after(runs[(fam, "base")]), snapshot_after(runs[(fam, "base2")])
    nondet = diff_keys(S_b, S_b2)
    old_struct = structure(runs[(fam, "base")], nondet) == structure(r6, nondet)
    old_argv = child_argv_match(child_calls(runs[(fam, "base")]), child_calls(r6))[0]
    w6 = run_worker(srcs["new"], mm / "S06" / "worker", argv, mutation=m6, mark=mark6)
    new_ok6 = "_error" not in w6 and all(same(W["base"], w6))
    hits6 = mark6.read_text().count("hit") if mark6.exists() else 0
    gate("MUTANT S06_worker_changes_learning_rate killed by the new gate, missed by the R1 gates",
         hits6 > 0 and r6["rc"] == 0 and old_struct and old_argv and not new_ok6,
         f"marker hits {hits6}; R1 structure gate {'PASS (missed)' if old_struct else 'FAIL'}, R1 child-argv gate "
         f"{'PASS (missed)' if old_argv else 'FAIL'}; new boundary gate {'PASS' if new_ok6 else 'FAIL'} "
         f"(params lr {(w6.get('boundary') or {}).get('params', {}).get('learning_rate')} vs base "
         f"{W['base']['boundary']['params'].get('learning_rate')})")


def winner_restored(run):
    """(ok, detail): after a compare run, every best_model.* file in the recorded winner's archive folder
    (compare_models/<S88>/<winner>/) is byte-identical at the top level (models/reinforcement/), as the
    restore at meta_prediction_optimizer_anti_overfit.py:327-334 intends. The winner is chosen from scores
    that vary run to run (children are unseeded), so top-level files are checked per run, not across runs."""
    d = run["dir"]
    fs = sorted((d / "diagnostics_outputs").glob("compare_models_summary_S88_*.json"))
    if len(fs) != 1:
        return False, f"{len(fs)} summaries"
    j = json.loads(fs[0].read_text())
    w = (j.get("winner_best_effort") or {}).get("model_type")
    arc = d / "models" / "reinforcement" / "compare_models" / j.get("run_id", "") / str(w)
    if not w or not arc.is_dir():
        return False, f"winner {w}; archive {arc.name} missing"
    names = sorted(p.name for p in arc.iterdir() if p.name.startswith("best_model"))
    bad = [n for n in names if (d / "models" / "reinforcement" / n).read_bytes() != (arc / n).read_bytes()]
    return (bool(names) and not bad), f"winner {w}: {len(names)} archived file(s) restored; differing {bad}"


def compare_part(out, srcs, s5, lot):
    """(4) WATCHER's Step 5 route: --compare-models --enable-diagnostics, base against new."""
    plan = [("base", "base", []), ("base2", "base", []), ("new_default", "new", []),
            ("new_legacy", "new", ["--diagnostics-level", "legacy"]),
            ("new_full", "new", ["--diagnostics-level", "full"])]
    R = {}
    for name, tk, extra in plan:
        R[name] = run_step5(srcs[tk], out / "compare" / name, s5, lot, "catboost", ["--compare-models"] + extra)
        print(f"  ran compare/{name}: rc {R[name]['rc']} in {R[name]['seconds']} s", flush=True)
    gate("CMP.all_runs_exit_0", all(r["rc"] == 0 for r in R.values()), str({n: r["rc"] for n, r in R.items()}))
    C = {n: child_calls(r) for n, r in R.items()}
    gate("CMP.control_base_twice_child_argv_equal", C["base"] == C["base2"] and len(C["base"]) == 4,
         f"{len(C['base'])} compare children (one per family)")
    for n in ("new_default", "new_legacy"):
        gate(f"CMP.{n}_child_argv_equal_base", C[n] == C["base"],
             f"{len(C[n])} children; first differing {next((i for i, (x, y) in enumerate(zip(C['base'], C[n])) if x != y), None)}")
    gate("CMP.nonvacuity_full_threads_flags_to_children",
         len(C["new_full"]) == 4 and all("--diagnostics-level" in c and "full" in c for c in C["new_full"]),
         "every child gets --diagnostics-level full")
    D = {n: compare_digest(r) for n, r in R.items()}
    gate("CMP.control_base_twice_summary_digest", D["base"] == D["base2"], json.dumps(D["base"])[:240])
    for n in ("new_default", "new_legacy"):
        gate(f"CMP.{n}_summary_digest_equal_base", D[n] == D["base"], json.dumps(D[n])[:160])
    S = {n: snapshot_after(r) for n, r in R.items()}
    nondet = diff_keys(S["base"], S["base2"])
    det = lambda s: {k: v for k, v in s.items() if k not in nondet}
    gate("CMP.control_base_twice", True, f"{len(nondet)} non-deterministic file(s): {nondet[:12]}")
    # the top-level best_model.* are the run's own winner's (restored); the winner varies run to run
    top = [k for k in nondet if k.startswith("models/reinforcement/best_model")]
    nondet_s = [k for k in nondet if k not in top]
    for n in ("base", "base2", "new_default", "new_legacy", "new_full"):
        gate(f"CMP.{n}_top_level_is_own_winner_restored", *winner_restored(R[n]))
    for n in ("new_default", "new_legacy"):
        dk = diff_keys(det(S["base"]), det(S[n]))
        gate(f"CMP.{n}_files_equal_base", not dk and set(S["base"]) == set(S[n]),
             f"{len(det(S[n]))} compared by content; differ {dk}; only-in-one {sorted(set(S['base']) ^ set(S[n]))[:6]}")
        sb, sn = structure(R["base"], nondet_s), structure(R[n], nondet_s)
        gate(f"CMP.{n}_structure_equal_base", sb == sn, f"{len(sb)} by structure; differ {diff_keys(sb, sn)}")
    ptr = "diagnostics_outputs/compare_models_summary_latest.json"
    gate("CMP.nonvacuity_full_writes_pointer", ptr in S["new_full"] and ptr not in S["new_legacy"] and ptr not in S["base"],
         "the CD-06 pointer exists at full only")
    hb = health(R["base"]["dir"], R["base"]["dir"], {}, label="cmp_hc_base")
    hn = health(R["new_legacy"]["dir"], R["base"]["dir"], {}, label="cmp_hc_new")
    gate("CMP.health_as_watcher_calls_it_equals_base", "_error" not in hb and hb == hn,
         f"base {hb.get('severity')}/{hb.get('action')}; new {hn.get('severity')}/{hn.get('action')}")
    # S07: the compare child gets an extra flag at legacy; the R1 single-family gates never reach this code
    m7 = ("meta_prediction_optimizer_anti_overfit.py",
          '            cmd.extend(_td.diagnostics_child_flags(args_dict.get("diagnostics_options")))\n',
          '            cmd.extend(_td.diagnostics_child_flags(args_dict.get("diagnostics_options")) + '
          '(["--gradient-probe-rows", "8"] if ' + HIT + ' else []))\n')
    mm = out / "mutants_r2" / "S07"
    mm.mkdir(parents=True)
    mark_old = mm / "mark_single"
    rb = run_step5(srcs["base"], mm / "single_base", s5, lot, "lightgbm", [])
    rs = run_step5(srcs["new"], mm / "single_mut", s5, lot, "lightgbm", ["--diagnostics-level", "legacy"], mark_old, mutation=m7)
    old_ok = (rs["rc"] == 0 and snapshot_after(rb) == snapshot_after(rs) and child_calls(rb) == child_calls(rs))
    hits_old = mark_old.read_text().count("hit") if mark_old.exists() else 0
    mark = mm / "mark"
    rc7 = run_step5(srcs["new"], mm / "compare_mut", s5, lot, "catboost", ["--compare-models", "--diagnostics-level", "legacy"],
                    mark, mutation=m7)
    hits = mark.read_text().count("hit") if mark.exists() else 0
    new_ok = rc7["rc"] == 0 and child_calls(rc7) == C["base"]
    # S08: the restore copies another family's artifacts to the top level (not the recorded winner). The family
    # is lightgbm, or the NN when lightgbm won: their archives always differ (a failed catboost child archives
    # the leftover xgboost files, so catboost would be a no-op when xgboost wins; seen in the first attempt).
    m8 = ("meta_prediction_optimizer_anti_overfit.py",
          '    winner = summary["winner_best_effort"].get("model_type")\n',
          '    winner = summary["winner_best_effort"].get("model_type")\n    if ' + HIT + ':\n'
          '        winner = "neural_net" if winner == "lightgbm" else "lightgbm"\n')
    mm8 = out / "mutants_r2" / "S08"
    mm8.mkdir(parents=True)
    mark8 = mm8 / "mark"
    r8 = run_step5(srcs["new"], mm8 / "compare_mut", s5, lot, "catboost", ["--compare-models", "--diagnostics-level", "legacy"],
                   mark8, mutation=m8)
    ok8, det8 = winner_restored(r8)
    hits8 = mark8.read_text().count("hit") if mark8.exists() else 0
    gate("MUTANT S08_restore_uses_another_family killed by the winner-restore gate",
         hits8 > 0 and r8["rc"] == 0 and not ok8, f"marker hits {hits8}; gate {'PASS' if ok8 else 'FAIL'}: {det8}")
    gate("MUTANT S07_compare_child_gets_a_flag_at_legacy killed by the compare gate, missed by the R1 gates",
         hits > 0 and old_ok and not new_ok,
         f"R1 single-family gates {'PASS (missed; marker hits ' + str(hits_old) + ', the code is not on that path)' if old_ok else 'FAIL'}; "
         f"compare gate {'PASS' if new_ok else 'FAIL'} (marker hits {hits})")


# ---------------------------------------------------------------- Q4 vacuity points (Beta X-CH14-DIAG R1 :91)
# The NN runs 2 trials x 2 folds, each a train_single_trial child, then one final child; the tree families
# train in-process (no child). The runs below use --trials 2 --k-folds 2.
EXPECTED_CHILD_CALLS = {"neural_net": 2 * 2 + 1, "lightgbm": 0, "xgboost": 0, "catboost": 0}


def child_count_ok(fam, calls):
    """(ok, detail): the run started exactly the expected number of children. An empty trace on the NN
    (capture lost, or no child started) is a failure, not 'equal to an empty base'."""
    want = EXPECTED_CHILD_CALLS[fam]
    return len(calls) == want, f"{len(calls)} child call(s), expected {want}"


def fixed_health_expectation(fam, legacy, fixed):
    """(ok, detail) for 'grading_rules fixed' on the base run's real report.
    Tree families wrote no report at legacy on this fixture: both rules must say absent/PROCEED (no change is
    expected and none is claimed). The NN wrote one: grade or action must change. A difference only in other
    result keys (grading_rules, report_source, ...) no longer counts."""
    g = lambda r: (r.get("severity"), r.get("action"))
    if "_error" in legacy or "_error" in fixed:
        return False, f"error: {legacy.get('_error') or fixed.get('_error')}"
    if fam != "neural_net":
        ok = g(legacy) == g(fixed) == ("absent", "PROCEED")
        return ok, f"no report: legacy {g(legacy)}, fixed {g(fixed)} (expected absent/PROCEED for both)"
    return g(legacy) != g(fixed), f"legacy {g(legacy)} -> fixed {g(fixed)} (grade or action must change)"


def vacuity_mutants(out, runs, srcs, s5, lot, families, fx):
    """One mutant per Q4 vacuity point that the R1 gate missed and the new gate catches.
    V1 (the base pin) is in tests/test_ch14_diag_wave1.py; its evidence is a separate run (READY note)."""
    mm = out / "mutants_q4"
    # V2: both child traces empty (capture lost). R1 child_argv_match([], []) said True.
    if "neural_net" in families:
        old = child_argv_match([], [])[0]
        new = child_count_ok("neural_net", [])[0]
        gate("MUTANT V2_child_traces_empty killed by the count gate, missed by the R1 gate",
             old and not new, f"R1 child-argv gate on two empty traces: {'PASS (missed)' if old else 'FAIL'}; "
                              f"count gate: {'PASS' if new else 'FAIL'}")
    # V3: at full, the inline tree report is written to the run root instead of diagnostics_outputs/.
    if "lightgbm" in families:
        m3 = ("meta_prediction_optimizer_anti_overfit.py",
              "            path = os.path.join('diagnostics_outputs', f'{self.model_type}_diagnostics.json')\n"
              "            diag.save(path)\n"
              "            _sh.copyfile(path, os.path.join('diagnostics_outputs', 'training_diagnostics.json'))\n",
              "            path = (f'{self.model_type}_diagnostics.json' if " + HIT + " else "
              "os.path.join('diagnostics_outputs', f'{self.model_type}_diagnostics.json'))\n"
              "            diag.save(path)\n"
              "            _sh.copyfile(path, os.path.join(os.path.dirname(path) or '.', 'training_diagnostics.json'))\n")
        mark = mm / "V3" / "mark"
        (mm / "V3").mkdir(parents=True)
        r = run_step5(srcs["new"], mm / "V3" / "run", s5, lot, "lightgbm", ["--diagnostics-level", "full"], mark, mutation=m3)
        Sb, Sb2, Sm = (snapshot_after(runs[("lightgbm", "base")]), snapshot_after(runs[("lightgbm", "base2")]),
                       snapshot_after(r))
        nondet = diff_keys(Sb, Sb2)
        dk = diff_keys({k: v for k, v in Sb.items() if k not in nondet}, {k: v for k, v in Sm.items() if k not in nondet})
        old = r["rc"] == 0 and bool(dk) and not diff_keys(
            {k: v for k, v in training_part(Sb).items() if k not in nondet},
            {k: v for k, v in training_part(Sm).items() if k not in nondet})
        new = bool([k for k in dk if k.startswith("diagnostics_outputs/")])
        hits = mark.read_text().count("hit") if mark.exists() else 0
        gate("MUTANT V3_full_report_written_outside_diagnostics_outputs killed by the new gate, missed by the R1 gate",
             hits > 0 and old and not new,
             f"marker hits {hits}; differing {dk[:4]}; R1 'any difference' gate {'PASS (missed)' if old else 'FAIL'}; "
             f"new 'a diagnostics_outputs/ report changed' gate {'PASS' if new else 'FAIL'}")
    # V4: the fixed path grades with the legacy rules. The R1 dict-inequality row still passes.
    if "neural_net" in families:
        m4 = ("training_health_check.py",
              "    g = grade_report(diag, policies, 'fixed')\n",
              "    g = grade_report(diag, policies, 'legacy' if " + HIT + " else 'fixed')\n")
        mark = mm / "V4" / "mark"
        (mm / "V4").mkdir(parents=True)
        mt = build_tree(mm / "V4" / "tree", srcs["new"])
        mutate(mt, *m4)
        base_dir = runs[("neural_net", "base")]["dir"]
        hn = health(runs[("neural_net", "new_legacy")]["dir"], base_dir, {}, label="v4_legacy")
        hf = health(mt, base_dir, {"grading_rules": "fixed"}, mark=mark, label="v4_fixed_mut")
        old = "_error" not in hf and hf != hn
        ffx = health(mt, fx, {"grading_rules": "fixed"}, mark=mark, label="v4_fixed_mut_fx")
        new_fx = (ffx.get("severity"), ffx.get("action")) == ("ok", "PROCEED")
        hits = mark.read_text().count("hit") if mark.exists() else 0
        gate("MUTANT V4_fixed_path_grades_with_legacy_rules killed by the new gate, missed by the R1 row",
             hits > 0 and old and not new_fx,
             f"marker hits {hits}; R1 dict-inequality row (NN real report) {'PASS (missed)' if old else 'FAIL'}; "
             f"new fixed-report expectation {'PASS' if new_fx else 'FAIL'} "
             f"(fixed report under mutant: {ffx.get('severity')}/{ffx.get('action')})")


# ---------------------------------------------------------------- Q4 negative control (item 2)
def negative_control(out, srcs, s5, lot):
    """CatBoost write-through control, entirely inside --out (Beta R3 Q5 :101). A disposable SOURCE ROOT under --out
    mirrors the worktree's top level by links, except catboost_info/, which is a real directory holding a sentinel.
    RED: the R1 tree policy ('legacy') built from that source root links its catboost_info/, CatBoost writes through
    it, and the snapshot of that directory changes; the R1 git-status view is unchanged (missed). GREEN: the new
    policy does not link it. Nothing is planted at the worktree root."""
    nc = out / "negative_control"
    src = nc / "source_root"
    src.mkdir(parents=True)
    for e in REPO.iterdir():
        if e.name in (".git", "data", "catboost_info"):
            continue
        os.symlink(e, src / e.name)
    target = src / "catboost_info"
    def plant():
        if target.exists():
            shutil.rmtree(target)
        target.mkdir()
        (target / "SENTINEL_ch14_negative_control.txt").write_text("planted inside --out by test_ch14_diag_land_defaults.py\n")
        (target / "time_left.tsv").write_text("SENTINEL\n")
    def status():
        out_ = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--ignored"],
                              capture_output=True, text=True).stdout
        return "\n".join(l for l in out_.splitlines() if not l[3:].startswith("data/"))
    results = {}
    for label, policy in (("red", "legacy"), ("green", "tracked")):
        plant()
        st0, iso0 = status(), isolation_snapshot([target])
        d = nc / label
        build_tree(d, srcs["new"], link_policy=policy, src_root=src)
        linked = (d / "catboost_info").is_symlink()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", S88_FORCE_LGBM_CPU="1", PYTHONDONTWRITEBYTECODE="1",
                   PYTHONPATH=str(d))
        argv = [sys.executable, "-B", str(REPO / "scripts/diagnostics/e2e_step5_shim.py"),
                "--split-seed", str(SPLIT_SEED), "--capture-dir", str(d / "shim"),
                "--script", str(d / "meta_prediction_optimizer_anti_overfit.py"), "--",
                "--survivors", str(s5), "--lottery-data", str(lot), "--model-type", "catboost",
                "--trials", "2", "--k-folds", "2", "--holdout-draws", "60", "--device", "cpu",
                "--output-dir", str(d / "models" / "reinforcement"), "--study-name", f"ch14land_nc_{label}",
                "--storage", f"sqlite:///{d / 'optuna.db'}", "--enable-diagnostics"]
        p = subprocess.run(argv, cwd=d, env=env, capture_output=True, text=True)
        (d / "cli.log").write_text(p.stdout + p.stderr)
        st1, iso1 = status(), isolation_snapshot([target])
        changed = sorted(k for k in set(iso0) | set(iso1) if iso0.get(k) != iso1.get(k))
        results[label] = {"rc": p.returncode, "linked": linked, "changed": changed, "status_same": st0 == st1}
    R_, G_ = results["red"], results["green"]
    gate("NC.red: the R1 tree policy writes through a catboost_info/ in a disposable source root inside --out; "
         "the snapshot is RED, the R1 git-status view is not",
         R_["rc"] == 0 and R_["linked"] and bool(R_["changed"]) and R_["status_same"],
         f"linked {R_['linked']}; files changed {R_['changed'][:3]}; R1 status view unchanged {R_['status_same']} (missed)")
    gate("NC.green: the new tree policy leaves that catboost_info/ byte-identical",
         G_["rc"] == 0 and not G_["linked"] and not G_["changed"],
         f"linked {G_['linked']}; files changed {G_['changed']}")
    gate("NC.nothing_planted_at_the_worktree_root", not (REPO / "catboost_info").exists(), str(REPO / "catboost_info"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--families", nargs="*", default=list(FAMILIES))
    ap.add_argument("--part", choices=("single", "compare"), default="single",
                    help="single: per-family Step 5 runs, the worker boundary and their mutants; "
                         "compare: --compare-models runs and S07 (two invocations keep each under 10 min)")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    # --out must be new, or hold only the caller's console log (so a `| tee <out>/console.log` stays inside --out)
    if not str(out).startswith(str(REPO / "data") + "/") or (out.exists() and set(os.listdir(out)) - {"console.log"}):
        raise SystemExit("--out must be a NEW directory under <worktree>/data/ (or hold only console.log)")
    status_before = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--ignored"],
                                   capture_output=True, text=True).stdout
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    redirect_env_roots(out)          # before any third-party import; inherited by every child (Beta R3 Q5)
    iso_roots = isolation_roots()
    iso_before = isolation_snapshot(iso_roots)
    bnd_before = boundary_snapshot(out)

    full = git("rev-parse", "--verify", f"{BASE_REV}^{{commit}}").decode().strip()
    base_src = {f: git("show", f"{BASE_REV}:{f}") for f in CODE_FILES}
    new_src = {f: (REPO / f).read_bytes() for f in CODE_FILES}
    gate("LAND.base_precondition",
         full == BASE_REV and len(BASE_REV) == 40
         and b"--diagnostics-level" not in base_src["meta_prediction_optimizer_anti_overfit.py"]
         and b"grading_rules" not in base_src["training_health_check.py"]
         and b"--diagnostics-level" in new_src["meta_prediction_optimizer_anti_overfit.py"]
         and b"--exclude-feature-groups" in base_src["meta_prediction_optimizer_anti_overfit.py"]
         and b"--exclude-feature-groups" in new_src["meta_prediction_optimizer_anti_overfit.py"],
         f"base {BASE_REV[:12]} lacks the branch's flags and has main's LL-3; new tree has both")
    changed = [f for f in CODE_FILES if base_src[f] != new_src[f]]
    gate("LAND.trees_differ_only_in_code_files", len(changed) == len(CODE_FILES), f"{len(changed)}/{len(CODE_FILES)} differ")

    s5, lot = make_fixture(out)
    srcs = {"base": base_src, "new": new_src}
    if a.part == "compare":
        compare_part(out, srcs, s5, lot)
        return finish(out, t0, status_before, [], iso_before, iso_roots, bnd_before=bnd_before)

    plan = [("base", "base", []), ("base2", "base", []), ("new_default", "new", []),
            ("new_legacy", "new", ["--diagnostics-level", "legacy"]),
            ("new_full", "new", ["--diagnostics-level", "full"])]
    runs = {}
    for fam in a.families:
        for name, tk, extra in plan:
            r = run_step5(srcs[tk], out / "runs" / fam / name, s5, lot, fam, extra)
            runs[(fam, name)] = r
            print(f"  ran {fam}/{name}: rc {r['rc']} in {r['seconds']} s", flush=True)

    summary_rows = []
    for fam in a.families:
        R = {n: runs[(fam, n)] for n, _, _ in plan}
        ok_rc = all(r["rc"] == 0 for r in R.values())
        gate(f"LAND.{fam}.all_runs_exit_0", ok_rc, str({n: r["rc"] for n, r in R.items()}))
        if not ok_rc:
            continue
        S = {n: snapshot_after(r) for n, r in R.items()}
        nondet = diff_keys(S["base"], S["base2"])
        det = lambda s: {k: v for k, v in s.items() if k not in nondet}
        gate(f"LAND.{fam}.control_base_twice", True, f"{len(nondet)} non-deterministic file(s) excluded: {nondet}")
        for n in ("new_default", "new_legacy"):
            dk = diff_keys(det(S["base"]), det(S[n]))
            gate(f"LAND.{fam}.{n}_equals_base", not dk and set(S["base"]) == set(S[n]),
                 f"{len(det(S[n]))} files compared byte/content-equal; differ: {dk}; "
                 f"only-in-one: {sorted(set(S['base']) ^ set(S[n]))}")
            if nondet:
                sb, sn = structure(R["base"], nondet), structure(R[n], nondet)
                gate(f"LAND.{fam}.{n}_structure_equals_base (non-deterministic files)", sb == sn,
                     f"{len(sb)} file(s) compared by key structure; differ: {diff_keys(sb, sn)}")
        # the children Step 5 starts (the NN trial worker): same command lines as BASE_REV at default and legacy
        cb = child_calls(R["base"])
        for n in ("base", "new_default", "new_legacy"):     # Q4 vacuity 2: an empty trace is not "equal"
            gate(f"LAND.{fam}.{n}_child_call_count", *child_count_ok(fam, child_calls(R[n])))
        for n in ("new_default", "new_legacy"):
            cn = child_calls(R[n])
            gate(f"LAND.{fam}.{n}_child_argv_equals_base", *child_argv_match(cb, cn))
        if cb:
            cf = child_calls(R["new_full"])
            gate(f"LAND.{fam}.nonvacuity_full_changes_child_argv", cf != cb and all(
                 set(x) <= set(y) for x, y in zip(cb[:-1], cf[:-1])),
                 f"full adds {next((sorted(set(y) - set(x))[:6] for x, y in zip(cb, cf) if x != y), [])}")
        tr = diff_keys(det(training_part(S["base"])), det(training_part(S["new_full"])))
        gate(f"LAND.{fam}.full_keeps_training_outputs", not tr, f"training files differ: {tr}")
        rep = diff_keys(det(S["base"]), det(S["new_full"]))
        if nondet:
            rep += [f"{k} (structure)" for k in diff_keys(structure(R["base"], nondet), structure(R["new_full"], nondet))]
        rep_d = [k for k in rep if k.startswith("diagnostics_outputs/")]   # Q4 vacuity 3: a REPORT must change
        gate(f"LAND.{fam}.nonvacuity_full_changes_reports", bool(rep_d),
             f"{len(rep_d)} file(s) under diagnostics_outputs/ differ: {rep_d[:6]} (all differing: {len(rep)})")
        # health check
        # same input (the base run's reports), two codes: base_rev's check and this landing's
        hb = health(R["base"]["dir"], R["base"]["dir"], {}, label="hc_base")
        hn = health(R["new_legacy"]["dir"], R["base"]["dir"], {}, label="hc_new_absent")
        hn2 = health(R["new_legacy"]["dir"], R["base"]["dir"], {"grading_rules": "legacy"}, label="hc_new_legacy")
        hf = health(R["new_legacy"]["dir"], R["base"]["dir"], {"grading_rules": "fixed"}, label="hc_new_fixed")
        gate(f"LAND.{fam}.health_legacy_equals_base", "_error" not in hb and hb == hn == hn2,
             f"base {hb.get('severity')}/{hb.get('action')}; new {hn.get('severity')}/{hn.get('action')}")
        # Q4 vacuity 4: grade/action, never whole dicts. Trees: absent/PROCEED under both rules is asserted.
        # The NN's real report changes run to run (CPU training is not reproducible), so whether 'fixed' changes
        # its grade is data-dependent (measured both ways): recorded, not gated. The deterministic claim is on
        # the fixed synthetic report below (exactly critical/RETRY -> ok/PROCEED).
        if fam != "neural_net":
            gate(f"LAND.{fam}.fixed_rules_health_expectation", *fixed_health_expectation(fam, hn, hf))
        else:   # Beta R3 Q6: the fixed rules must grade the real NN report validly (either grade), never error out
            iss = hf.get("issues") or []
            gate(f"LAND.{fam}.fixed_rules_grade_real_report_validly",
                 "_error" not in hf and hf.get("severity") in ("ok", "warning", "critical")
                 and not any(str(x).startswith("Health check module error") for x in iss),
                 f"fixed {hf.get('severity')}/{hf.get('action')}; module-error issues "
                 f"{[x for x in iss if str(x).startswith('Health check module error')]}")
        summary_rows.append({"family": fam, "nondeterministic_excluded": nondet,
                             "health_legacy_to_fixed": [hn.get("severity"), hn.get("action"), hf.get("severity"), hf.get("action")],
                             "health": {"base": hb.get("severity"), "new_legacy": hn.get("severity"),
                                        "new_fixed": hf.get("severity")},
                             "seconds": {n: r["seconds"] for n, r in R.items()}})

    # ---------------------------------------------------------------- mutants (lightgbm and neural_net)
    mutants = [
        ("S01_step5_legacy_treated_as_full", "meta_prediction_optimizer_anti_overfit.py",
         "            getattr(self, '_cli_diagnostics_level', 'legacy') == 'full'\n",
         f"            ({HIT} or getattr(self, '_cli_diagnostics_level', 'legacy') == 'full')\n",
         "lightgbm", "new_legacy_equals_base"),
        ("S02_step5_training_changed", "meta_prediction_optimizer_anti_overfit.py",
         "            'device': 'cpu' if os.environ.get('S88_FORCE_LGBM_CPU') == '1' else ('gpu' if 'cuda' in self.device else 'cpu'),\n            'random_state': 42,\n",
         f"            'device': 'cpu' if os.environ.get('S88_FORCE_LGBM_CPU') == '1' else ('gpu' if 'cuda' in self.device else 'cpu'),\n            'random_state': 42 + (1 if {HIT} else 0),\n",
         "lightgbm", "new_legacy_equals_base"),
        ("S03_health_legacy_silently_fixed", "training_health_check.py",
         "    best_round = training_summary.get('best_round', 0)\n",
         f"    best_round = {HIT} and training_summary.get('best_val_round', 0)\n",
         "fixed_report", "health_fixed_report_equals_base"),
        ("S04_nn_child_gets_a_flag_at_legacy", "meta_prediction_optimizer_anti_overfit.py",
         '                cmd.append("--enable-diagnostics")\n                if self._ch14_full():  # [CH14-DIAG waves 1-2]\n',
         '                cmd.append("--enable-diagnostics")\n                cmd.extend(["--diagnostics-level", "legacy"] if '
         + HIT + ' else [])\n                if self._ch14_full():  # [CH14-DIAG waves 1-2]\n',
         "neural_net", "child_argv_equals_base"),
    ]
    # ---------------------------------------------------------------- the health check on a FIXED report
    # (the NN's real reports change run to run; this one does not). Shape: a schema-1.1.0 NN report as
    # training_diagnostics writes it (best epoch in best_val_round, no best_round key: chapter-14 B1/CD-01).
    fx = out / "fixed_report"
    (fx / "diagnostics_outputs").mkdir(parents=True)
    rep_fx = {"status": "complete", "model_type": "neural_net", "schema_version": "1.1.0",
              "generated_at": "2026-11-25T00:00:00",
              "training_summary": {"rounds_captured": 17, "final_train_loss": 0.90, "final_val_loss": 0.95,
                                   "best_val_loss": 0.94, "best_val_round": 14, "overfit_gap": 0.05},
              "diagnosis": {"severity": "ok", "issues": [], "suggested_fixes": [], "confidence": 0.9},
              "model_specific": {"gradient_health": {"vanishing": False, "exploding": False, "dead_neuron_pct": 0.0,
                                                     "gradient_spread_ratio": 1.0},
                                 "layer_health": {"network.0": {"dead_pct": 0.0, "dead_trend": "stable",
                                                                "gradient_norm": 0.01}}},
              "round_data_truncated": False, "round_data_sample": []}
    for fn in ("training_diagnostics.json", "neural_net_diagnostics.json"):
        (fx / "diagnostics_outputs" / fn).write_text(json.dumps(rep_fx, indent=2))
    code_base = build_tree(out / "hc_code_base", base_src)
    code_new = build_tree(out / "hc_code_new", new_src)
    fb_ = health(code_base, fx, {}, label="fx_base")
    fn_ = health(code_new, fx, {}, label="fx_new_absent")
    fn2 = health(code_new, fx, {"grading_rules": "legacy"}, label="fx_new_legacy")
    ff = health(code_new, fx, {"grading_rules": "fixed"}, label="fx_new_fixed")
    gate("LAND.health_fixed_report_equals_base", "_error" not in fb_ and fb_ == fn_ == fn2,
         f"base {fb_.get('severity')}/{fb_.get('action')}; new {fn_.get('severity')}/{fn_.get('action')}")
    gate("LAND.nonvacuity_fixed_rules_on_fixed_report",
         "_error" not in ff and (fn_.get("severity"), fn_.get("action")) == ("critical", "RETRY")
         and (ff.get("severity"), ff.get("action")) == ("ok", "PROCEED"),
         f"expected legacy critical/RETRY -> fixed ok/PROCEED; got {fn_.get('severity')}/{fn_.get('action')} -> "
         f"{ff.get('severity')}/{ff.get('action')}")

    for mid, rel, old, new, fam, target in mutants:
        if fam != "fixed_report" and fam not in a.families:
            continue
        mark = out / "mutants" / mid / "mark"
        (out / "mutants" / mid).mkdir(parents=True)
        if target == "health_fixed_report_equals_base":
            mt = build_tree(out / "mutants" / mid / "tree", new_src)
            mutate(mt, rel, old, new)
            hm = health(mt, fx, {}, mark=mark, label=f"fx_{mid}")
            hits = mark.read_text().count("hit") if mark.exists() else 0
            gate(f"MUTANT {mid} killed", hits > 0 and hm != fb_,
                 f"target fixed_report.health; marker hits {hits}; base {fb_.get('severity')} vs mutant {hm.get('severity')}")
            continue
        base_dir = runs[(fam, "base")]["dir"]
        S_base = snapshot_after(runs[(fam, "base")])
        nondet = diff_keys(S_base, snapshot_after(runs[(fam, "base2")]))
        if target == "child_argv_equals_base":
            r = run_step5(new_src, out / "mutants" / mid / "run", s5, lot, fam, ["--diagnostics-level", "legacy"],
                          mark, mutation=(rel, old, new))
            cb, cm = child_calls(runs[(fam, "base")]), child_calls(r)
            gate_ok = r["rc"] == 0 and child_argv_match(cb, cm)[0]
            dk = [f"{len(cm)} child calls; extra args {sorted(set(cm[0]) - set(cb[0])) if cm and cb else []}"]
        elif target == "new_legacy_equals_base":
            r = run_step5(new_src, out / "mutants" / mid / "run", s5, lot, fam, ["--diagnostics-level", "legacy"],
                          mark, mutation=(rel, old, new))
            Sm = snapshot_after(r)
            dk = diff_keys({k: v for k, v in S_base.items() if k not in nondet},
                           {k: v for k, v in Sm.items() if k not in nondet})
            gate_ok = r["rc"] == 0 and not dk and set(S_base) == set(Sm)
        else:
            mt = build_tree(out / "mutants" / mid / "tree", new_src)
            mutate(mt, rel, old, new)
            hb = health(base_dir, base_dir, {}, label=f"hc_{mid}_base")
            hm = health(mt, base_dir, {}, mark=mark, label=f"hc_{mid}")
            gate_ok = hb == hm
            dk = [f"base {hb.get('severity')}/{len(hb.get('issues') or [])} issues vs mutant "
                  f"{hm.get('severity')}/{len(hm.get('issues') or [])} issues; differing keys "
                  f"{sorted(k for k in set(hb) | set(hm) if hb.get(k) != hm.get(k))}"]
        hits = mark.read_text().count("hit") if mark.exists() else 0
        ran = hits > 0
        gate(f"MUTANT {mid} killed", ran and not gate_ok,
             f"target {fam}.{target}; marker hits {hits}; gate under mutant {'PASS' if gate_ok else 'FAIL'}; {str(dk)[:200]}")

    single_part_extras(out, runs, srcs, s5, lot, a.families)
    vacuity_mutants(out, runs, srcs, s5, lot, a.families, fx)
    return finish(out, t0, status_before, summary_rows, iso_before, iso_roots,
                  after_gates=lambda: negative_control(out, srcs, s5, lot), bnd_before=bnd_before)


def finish(out, t0, status_before, summary_rows, iso_before=None, iso_roots=None, after_gates=None, bnd_before=None):
    if bnd_before is not None:
        ch = boundary_diff(bnd_before, boundary_snapshot(out))
        gate("BOUNDARY.nothing_outside_out_changed (data/ minus --out, the tree outside data/, git status, "
             "/tmp/matplotlib-*, user matplotlib dirs)", not ch,
             f"{len(bnd_before['data'])} data/ files, {len(bnd_before['tree'])} tree files watched; changes {ch[:6]}")
        boundary_negative_control(out)
    if iso_before is not None:
        roots = sorted(set(iso_roots) | set(isolation_roots()))
        iso_after = isolation_snapshot(roots)
        changed = sorted(k for k in set(iso_before) | set(iso_after) if iso_before.get(k) != iso_after.get(k))
        gate("LAND.no_existing_dir_outside_out_changed (catboost_info: worktree, its parents, main checkout)",
             not changed, f"{len(roots)} root(s) {[str(r) for r in roots][:4]}; {len(iso_before)} entries; changed {changed[:6]}")
    status_after = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--ignored"],
                                  capture_output=True, text=True).stdout
    new_lines = sorted(set(status_after.splitlines()) - set(status_before.splitlines()))
    gate("LAND.wrote_nothing_outside_out",
         all(l.endswith("data/") or l.startswith("!! data/") or l == "?? data/" for l in new_lines),
         f"new git-status lines: {new_lines}")

    if after_gates:
        after_gates()
    summ = {"base_rev": BASE_REV, "split_seed": SPLIT_SEED, "gates": len(RESULTS),
            "passed": sum(r["ok"] for r in RESULTS), "failed": [r["gate"] for r in RESULTS if not r["ok"]],
            "mutants": sum(1 for r in RESULTS if r["gate"].startswith("MUTANT")),
            "mutants_killed": sum(1 for r in RESULTS if r["gate"].startswith("MUTANT") and r["ok"]),
            "wall_seconds": round(time.time() - t0, 1),
            "parent_peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "children_peak_rss_mb": round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024, 1),
            "families": summary_rows, "results": RESULTS}
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    print(f"\n{summ['passed']}/{summ['gates']} gates pass; mutants killed {summ['mutants_killed']}/{summ['mutants']}; "
          f"wall {summ['wall_seconds']} s; child peak RSS {summ['children_peak_rss_mb']} MB", flush=True)
    return 0 if not summ["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
