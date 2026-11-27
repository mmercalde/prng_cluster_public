#!/usr/bin/env python3
"""
CH14-DIAG wave 1 tests (2026-11-23). CPU only, synthetic fixtures only.

Items: CD-00 offline re-grade, CD-01 early-stop key, CD-02 spread key (reader + producer),
CD-03 NN eval-mode gradient probe, CD-04 feature names, CD-05 tree reports (worker Booster path
and mpo inline final path), CD-06 compare-models winner (pointer writer + reader), CD-13 column
classes, and "default unchanged" gates against the code at HEAD_REV, the main commit this
lands on (7f4505a, full sha below; landing re-fit 2026-11-25, was 41ad509).

Every credited assertion carries a [CREDIT:<tag>] marker; tests/ch14_wave1_mutants.py applies
each mutant, proves it ran (marker file) and that the run failed on its credited tag.

Run (from the worktree root):
    CUDA_VISIBLE_DEVICES= python3 -m pytest -q tests/test_ch14_diag_wave1.py \
        --basetemp=data/ch14/pytest_tmp -p no:cacheprovider
Code under test is imported from $CH14_CODE_ROOT (default: the worktree root). Everything the
suite writes is under pytest's --basetemp (the base-revision copies included).
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # no GPU, set before any ML import
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import contextlib
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

WORKTREE = Path(__file__).resolve().parents[1]
CODE_ROOT = Path(os.environ.get("CH14_CODE_ROOT", str(WORKTREE))).resolve()
# main at landing (full sha, never HEAD or a merge base). CH14_BASE_REV exists only so the
# precondition gate can be shown to fail on a revision that already holds this work.
HEAD_REV = os.environ.get("CH14_BASE_REV", "7f4505aaccebdd284393eb3565e75d09d1c0f3b4")
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(1, str(Path(__file__).resolve().parent))

import training_diagnostics as td          # noqa: E402
import training_health_check as thc        # noqa: E402
import train_single_trial as tst           # noqa: E402

PY = sys.executable
SUB_ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(CODE_ROOT),
               PYTHONDONTWRITEBYTECODE="1")


def credit(tag):
    return f"[CREDIT:{tag}]"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def head_dir(tmp_path_factory):
    """Code at HEAD_REV for the default-unchanged gates (git show; never checked out), under --basetemp."""
    HEAD_DIR = tmp_path_factory.mktemp(f"head_{HEAD_REV[:7]}")
    for fn in ("training_health_check.py", "training_diagnostics.py", "train_single_trial.py"):
        out = subprocess.run(["git", "-C", str(WORKTREE), "show", f"{HEAD_REV}:{fn}"],
                             capture_output=True, check=True).stdout
        (HEAD_DIR / fn).write_bytes(out)
    link = HEAD_DIR / "models"
    if not link.exists():
        link.symlink_to(WORKTREE / "models")
    _base_precondition(HEAD_DIR)          # every identity test fails on a wrong pin (Beta R1 Q4, vacuity 1)
    return HEAD_DIR


def _base_precondition(head_dir):
    """The default-unchanged gates compare against HEAD_REV, the main commit this lands on. They are
    vacuous if the pin is not a full sha, if that revision already holds this work, or if it is an OLDER
    main that lacks what main has since landed in the same files (Beta X-CH14-DIAG R1 :91: the full
    41ad509 sha passed the R1 version of this check). The marker for 'is main at or after the LL
    landing' is LL-3's --exclude-feature-groups in the optimizer (landed 7cac105)."""
    full = subprocess.run(["git", "-C", str(WORKTREE), "rev-parse", "--verify", f"{HEAD_REV}^{{commit}}"],
                          capture_output=True, text=True, check=True).stdout.strip()
    assert len(HEAD_REV) == 40 and full == HEAD_REV, credit("BASE_PRECONDITION")
    td_src = (head_dir / "training_diagnostics.py").read_text()
    thc_src = (head_dir / "training_health_check.py").read_text()
    tst_src = (head_dir / "train_single_trial.py").read_text()
    assert "DIAGNOSTICS_LEVELS" not in td_src and "grading_rules" not in thc_src \
        and "--diagnostics-level" not in tst_src, credit("BASE_PRECONDITION")
    mpo = subprocess.run(["git", "-C", str(WORKTREE), "show", f"{HEAD_REV}:meta_prediction_optimizer_anti_overfit.py"],
                         capture_output=True, check=True).stdout
    assert b"--exclude-feature-groups" in mpo and b"--diagnostics-level" not in mpo, credit("BASE_PRECONDITION")


def test_base_revision_precondition(head_dir):
    """Explicit test of the same check the head_dir fixture enforces."""
    _base_precondition(head_dir)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def nn_report(best_val_round=14, rounds=17, spread=1.0, producer="ok", train=0.90, val=0.95,
              spread_key="gradient_health"):
    """Report shaped like a real schema-1.1.0 NN file (keys as td writes them)."""
    ms = {"gradient_health": {"vanishing": False, "exploding": False, "dead_neuron_pct": 0.0,
                              "gradient_spread_ratio": 1.0},
          "layer_health": {"network.0": {"dead_pct": 0.0, "dead_trend": "stable", "gradient_norm": 0.01}}}
    if spread_key == "gradient_health":
        ms["gradient_health"]["gradient_spread_ratio"] = spread
    else:
        ms["feature_gradient_spread"] = spread
    return {"status": "complete",
            "training_summary": {"rounds_captured": rounds, "final_train_loss": train,
                                 "final_val_loss": val, "best_val_loss": val - 0.01,
                                 "best_val_round": best_val_round, "overfit_gap": val - train},
            "diagnosis": {"severity": producer, "issues": [], "suggested_fixes": [], "confidence": 0.9},
            "model_specific": ms, "round_data_truncated": False, "round_data_sample": [],
            "model_type": "neural_net", "generated_at": "2026-11-23T10:00:00", "schema_version": "1.1.0"}


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
    return path


def tree_data(n=360, p=12, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float32)
    X[:, p - 1] = 7.0                                    # a constant column
    y = (1.5 * X[:, 0] - X[:, 1] + 0.7 * rng.normal(size=n)).astype(np.float32)
    names = [f"col_{i:02d}" for i in range(p - 1)] + ["konst_global"]
    return X[:240], y[:240], X[240:], y[240:], names


from ch14_cpu_patch import force_cpu_trees, TREE_PARAMS  # noqa: E402  (no code-root imports)


# ─────────────────────────────────────────────────────────────────────────────
# CD-01 / CD-02 reader / default unchanged (health check)
# ─────────────────────────────────────────────────────────────────────────────

def _run_check(mod, cwd, monkeypatch, files, **kw):
    Path(cwd).mkdir(parents=True, exist_ok=True)
    for rel, obj in files.items():
        if obj is None:
            continue
        p = Path(cwd) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(obj if isinstance(obj, str) else json.dumps(obj))
    monkeypatch.chdir(cwd)
    # [flake fix 2026-11-27, W3 fix pass 1] the same fix as tests/test_benching_switch.py:313-325 (aefdd74, owner
    # approved 2026-11-26 11:34): _archive_diagnostics names files <type>_<YYYYmmdd_HHMMSS>.json, so two checks
    # of one type in the same wall-clock second overwrite one archive, and head and new trees differed only by
    # where the second boundary fell. Each fresh archive is renamed to a unique, ordered name right after it is
    # written, so every archive is kept and compared; file contents are untouched. _archive_diagnostics unchanged.
    orig_archive = getattr(mod._archive_diagnostics, "_orig_archive", mod._archive_diagnostics)

    def archive_kept(*a, **k):
        out = orig_archive(*a, **k)
        h = Path(cwd) / "diagnostics_outputs" / "history"
        n = len([p for p in h.glob("*.json") if not re.fullmatch(r".+_\d{8}_\d{6}\.json", p.name)])
        for p in sorted(h.glob("*.json")):
            if re.fullmatch(r".+_\d{8}_\d{6}\.json", p.name):
                n += 1
                p.rename(p.with_name(f"{p.stem.rsplit('_', 2)[0]}_{n:04d}.json"))
        return out
    archive_kept._orig_archive = orig_archive
    monkeypatch.setattr(mod, "_archive_diagnostics", archive_kept)
    res = mod.check_training_health(**kw)
    hist = sorted((Path(cwd) / "diagnostics_outputs" / "history").glob("*.json"))
    reg_p = Path(cwd) / "diagnostics_outputs" / "model_skip_registry.json"
    reg = json.loads(reg_p.read_text()) if reg_p.exists() else {}
    archived = [{k: v for k, v in json.loads(h.read_text()).items() if k != "archived_at"} for h in hist]
    return res, {m: e.get("consecutive_critical") for m, e in reg.items()}, archived


LEGACY_CASES = {
    "late_best": {"diagnostics_outputs/training_diagnostics.json": nn_report(14, 17)},
    "early_best": {"diagnostics_outputs/training_diagnostics.json": nn_report(1, 17)},
    "spread_old_key": {"diagnostics_outputs/training_diagnostics.json": nn_report(14, 17, 5000, spread_key="old")},
    "multi": {"diagnostics_outputs/training_diagnostics.json": {
        "models": {"neural_net": nn_report(1, 17), "catboost": dict(nn_report(14, 17), model_type="catboost")},
        "comparison": {"winner": "catboost"}}},
    "absent": {},
    "unreadable": {"diagnostics_outputs/training_diagnostics.json": "{not json"},
    "fixture_best_round": {"diagnostics_outputs/training_diagnostics.json": {
        "model_type": "neural_net", "training_summary": {"rounds_captured": 100, "best_round": 15,
                                                          "final_train_loss": 0.01, "final_val_loss": 0.02},
        "diagnosis": {"severity": "warning", "issues": ["x"], "suggested_fixes": []}}},
}


@pytest.mark.parametrize("case", sorted(LEGACY_CASES))
def test_legacy_rules_identical_to_base(case, head_dir, tmp_path, monkeypatch):
    head = load_module("thc_head_base", head_dir / "training_health_check.py")
    a = _run_check(head, tmp_path / "head", monkeypatch, LEGACY_CASES[case])
    b = _run_check(thc, tmp_path / "new", monkeypatch, LEGACY_CASES[case])
    # twice in a row, so the skip registry path (RETRY -> RETRY) is exercised too
    a2 = _run_check(head, tmp_path / "head", monkeypatch, {})
    b2 = _run_check(thc, tmp_path / "new", monkeypatch, {})
    assert (a, a2) == (b, b2), credit("LEGACY_IDENTITY")


def test_cd01_fixed_early_stop_key(tmp_path, monkeypatch):
    pol = {"grading_rules": "fixed"}
    healthy = thc.grade_report(nn_report(14, 17), pol)
    early = thc.grade_report(nn_report(1, 17), pol)
    legacy = thc.grade_report(nn_report(14, 17), {}, "legacy")
    assert legacy["severity"] == "critical"                      # the defect, reproduced
    assert healthy["severity"] == "ok", credit("CD01_OK")
    assert healthy["metrics"]["best_round_key"] == "best_val_round", credit("CD01_KEY")
    assert early["severity"] == "critical", credit("CD01_CRIT")
    assert any("Early stop ratio 0.06" in i for i in early["issues"]), credit("CD01_CRIT")
    base1 = thc.grade_report(nn_report(1, 17), dict(pol, early_stop_round_base=1))
    assert abs(base1["metrics"]["early_stop_ratio"] - 2 / 17) < 1e-12
    empty = thc.grade_report(dict(nn_report(), status="partial", training_summary={}), pol)
    assert empty["severity"] == "absent", credit("CD01_EMPTY")
    assert thc.grade_report(dict(nn_report(), status="partial", training_summary={}), {}, "legacy")["severity"] == "critical"
    short = thc.grade_report(nn_report(0, 1), pol)
    assert short["severity"] == "ok" and short["metrics"]["early_stop_ratio"] is None
    # through check_training_health (fixed via policies file) the action follows
    write_json(tmp_path / "watcher_policies.json", {"training_diagnostics": pol})
    write_json(tmp_path / "diagnostics_outputs/training_diagnostics.json", nn_report(14, 17))
    monkeypatch.chdir(tmp_path)
    r = thc.check_training_health()
    assert r["action"] == "PROCEED" and r["grading_rules"] == "fixed", credit("CD01_OK")
    with pytest.raises(ValueError):
        thc._grading_settings({"grading_rules": "bogus"})


def test_cd02_reader_spread_key():
    pol = {"grading_rules": "fixed"}
    crit = thc.grade_report(nn_report(14, 17, spread=5000.0), pol)
    ok = thc.grade_report(nn_report(14, 17, spread=50.0), pol)
    warn = thc.grade_report(nn_report(14, 17, spread=500.0), pol)
    assert crit["severity"] == "critical", credit("CD02_CRIT")
    assert any("gradient spread 5000x" in i for i in crit["issues"]), credit("CD02_CRIT")
    assert ok["severity"] == "ok" and warn["severity"] == "warning"
    legacy = thc.grade_report(nn_report(14, 17, spread=5000.0), {}, "legacy")
    assert not any("spread" in i for i in legacy["issues"])      # the defect: never fires


# ─────────────────────────────────────────────────────────────────────────────
# CD-06 winner: reader and pointer writer
# ─────────────────────────────────────────────────────────────────────────────

def _compare_dir(tmp_path, winner="catboost"):
    d = tmp_path / "diagnostics_outputs"
    write_json(d / "training_diagnostics.json", nn_report(1, 17))           # NN wrote last, critical
    write_json(d / "neural_net_diagnostics.json", nn_report(1, 17))
    write_json(d / "catboost_diagnostics.json", dict(nn_report(14, 17), model_type="catboost"))
    write_json(d / thc.COMPARE_POINTER_FILENAME, {
        "run_id": "S88_20261123_120000", "models": {"neural_net": {}, "catboost": {}, "xgboost": {}},
        "winner_best_effort": {"model_type": winner, "score": 0.1}})
    return d


def test_cd06_reader_grades_winner(tmp_path, monkeypatch):
    _compare_dir(tmp_path)
    write_json(tmp_path / "watcher_policies.json", {"training_diagnostics": {"grading_rules": "fixed"}})
    monkeypatch.chdir(tmp_path)
    r = thc.check_training_health()
    assert r["model_type"] == "catboost", credit("CD06_WINNER")
    assert r["report_source"] == "compare_pointer_winner_report", credit("CD06_WINNER")
    assert r["per_family"]["neural_net"]["severity"] == "critical"
    assert r["per_family"]["xgboost"]["severity"] == "absent"
    assert r["severity"] == "warning" and r["action"] == "PROCEED_WITH_NOTE"   # winner ok, NN not
    # legacy on the same directory grades the NN file
    write_json(tmp_path / "watcher_policies.json", {"training_diagnostics": {}})
    rl = thc.check_training_health()
    assert rl["model_type"] == "neural_net" and rl["severity"] == "critical"


def test_cd06_all_families_scope(tmp_path, monkeypatch):
    _compare_dir(tmp_path)
    write_json(tmp_path / "watcher_policies.json",
               {"training_diagnostics": {"grading_rules": "fixed", "grade_scope": "all_families"}})
    monkeypatch.chdir(tmp_path)
    r = thc.check_training_health()
    assert r["severity"] == "critical" and r["model_type"] == "neural_net", credit("CD06_ALL")
    assert r["action"] == "RETRY"


def test_cd06_winner_report_absent_is_absent_not_canonical(tmp_path, monkeypatch):
    _compare_dir(tmp_path, winner="lightgbm")
    write_json(tmp_path / "watcher_policies.json", {"training_diagnostics": {"grading_rules": "fixed"}})
    monkeypatch.chdir(tmp_path)
    r = thc.check_training_health()
    assert r["severity"] == "absent" and r["model_type"] == "lightgbm" and r["action"] == "PROCEED", \
        credit("CD06_ABSENT")


@pytest.mark.parametrize("level", ["legacy", "full"])
def test_cd06_pointer_writer(level, tmp_path, monkeypatch):
    import meta_prediction_optimizer_anti_overfit as mpo
    calls = []

    def fake_run(cmd, env=None, stdout=None, stderr=None, **kw):
        calls.append(list(cmd))
        m = cmd[cmd.index("--model-type") + 1]
        write_json(Path("models/reinforcement/best_model.meta.json"),
                   {"training_metrics": {"r2": {"neural_net": 0.1, "lightgbm": 0.2,
                                                 "xgboost": 0.3, "catboost": 0.4}[m]}})
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subprocess, "run", fake_run)
    rc = mpo._s88_run_compare_models({"survivors": "s.json", "lottery_data": "l.json", "trials": 1,
                                      "enable_diagnostics": True, "diagnostics_level": level})
    assert rc == 0
    per_run = list(Path("diagnostics_outputs").glob("compare_models_summary_S88_*.json"))
    ptr = Path("diagnostics_outputs") / "compare_models_summary_latest.json"
    assert len(per_run) == 1
    if level == "full":
        assert ptr.exists(), credit("CD06_POINTER")
        assert thc._winner_from_summary(json.loads(ptr.read_text())) == "catboost", credit("CD06_POINTER")
        assert calls and all("--diagnostics-level" in c and c[c.index("--diagnostics-level") + 1] == "full"
                             for c in calls), credit("CD06_CHILD_FLAG")
    else:
        assert not ptr.exists()
        assert all("--diagnostics-level" not in c for c in calls)


# ─────────────────────────────────────────────────────────────────────────────
# CD-13 column classes
# ─────────────────────────────────────────────────────────────────────────────

def test_cd13_column_classes():
    rng = np.random.default_rng(0)
    n, groups = 60, np.repeat(np.arange(6), 10)
    X = np.column_stack([
        np.full(n, 3.0),                           # constant
        np.full(n, 9.0),                           # constant, declared run-level
        np.repeat(rng.normal(size=6), 10),         # one value per trial
        rng.normal(size=n),                        # local
        np.full(n, np.nan),                        # all-NaN -> constant
    ])
    names = ["c", "run_c", "trial_c", "loc", "nan_c"]
    cc = td.classify_columns(X, names, groups=groups, run_level_names=["run_c"])
    assert cc["by_feature"] == {"c": "constant", "run_c": "run", "trial_c": "trial", "loc": "local",
                                "nan_c": "constant"}, credit("CD13_CLASSES")
    assert cc["counts"] == {"constant": 2, "run": 1, "trial": 1, "local": 1} and cc["non_local"] == 4
    no_groups = td.classify_columns(X, names)
    assert no_groups["trial_class"] == "unavailable" and no_groups["by_feature"]["trial_c"] == "local"
    share = td.attribution_share_by_class([1, 0, 1, 2, 0], names, cc["by_feature"])
    assert share["constant"] == 0.25 and share["trial"] == 0.25 and share["local"] == 0.5


# ─────────────────────────────────────────────────────────────────────────────
# CD-03 / CD-02 producer / CD-04 (NN)
# ─────────────────────────────────────────────────────────────────────────────

NN_PARAMS = {"hidden_layers": [16, 8], "epochs": 12, "early_stopping_patience": 50, "batch_size": 64}


def _nn_data():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 8)).astype(np.float32)
    X[:, 7] = 2.0
    y = (X[:, 0] + 0.3 * rng.normal(size=400)).astype(np.float32)
    return X, y


def _train_nn(tmp, tag, enable, opts, mode):
    import torch
    X, y = _nn_data()
    torch.manual_seed(0)
    r = tst.train_neural_net(X[:300], y[:300], X[300:], y[300:], dict(NN_PARAMS),
                             save_path=str(tmp / f"{tag}.pth"), enable_diagnostics=enable,
                             normalize_features=True, use_leaky_relu=True, batch_mode=mode,
                             diagnostics_options=opts)
    sd = torch.load(str(tmp / f"{tag}.pth"), weights_only=False)["state_dict"]
    rep = tmp / "diagnostics_outputs" / "neural_net_diagnostics.json"
    return r, sd, (json.loads(rep.read_text()) if rep.exists() else None)


@pytest.mark.parametrize("mode", ["full", "mini"])
def test_cd03_nn_bit_identical_and_probe(mode, tmp_path, monkeypatch):
    import torch
    monkeypatch.chdir(tmp_path)
    names = [f"n{i}" for i in range(8)]
    opts = {"level": "full", "feature_names": names, "names_source": "npz", "gradient_probe_rows": 64}
    r_off, sd_off, _ = _train_nn(tmp_path, "off", False, None, mode)
    r_leg, sd_leg, rep_leg = _train_nn(tmp_path, "leg", True, None, mode)
    r_full, sd_full, rep = _train_nn(tmp_path, "full", True, opts, mode)
    same = all(torch.equal(sd_off[k], sd_full[k]) for k in sd_off) and \
        all(torch.equal(sd_off[k], sd_leg[k]) for k in sd_off)
    assert same, credit("CD03_BITID")
    keys = ("train_mse", "val_mse", "r2", "best_iteration")
    assert [r_off[k] for k in keys] == [r_full[k] for k in keys] == [r_leg[k] for k in keys], credit("CD03_BITID")
    fg = rep["model_specific"].get("feature_gradients")
    assert fg and fg["source"] == "eval_probe", credit("CD03_GRADS")
    assert {e["feature"] for e in fg["top_10"]} <= set(names), credit("CD03_GRADS")
    assert rep["model_specific"]["gradient_health"]["gradient_spread_ratio"] != 1.0, credit("CD03_GRADS")
    # CD-02 producer: 12 epochs, detail every 5 -> last detail round 10, last round 11
    assert fg["round"] == 10, credit("CD02_PRODUCER")
    assert rep["feature_names_source"] == "npz" and rep["column_classes"]["by_feature"]["n7"] == "constant"
    assert "attribution_share_by_class" in rep
    assert "feature_gradients" not in rep_leg["model_specific"]       # legacy unchanged


def test_cd02_producer_last_round_with_spread():
    d = td.NNDiagnostics(diagnostics_level="full")
    d._round_data = [{"round": 0, "feature_gradients": {"spread_ratio": 77.0, "top_10": []}},
                     {"round": 1}, {"round": 2}]
    assert d._get_model_specific()["gradient_health"]["gradient_spread_ratio"] == 77.0, credit("CD02_PRODUCER")
    legacy = td.NNDiagnostics()
    legacy._round_data = list(d._round_data)
    assert legacy._get_model_specific()["gradient_health"]["gradient_spread_ratio"] == 1.0


def _write_npz(path, X, y, names=None):
    kw = dict(X_train=X[:300], y_train=y[:300], X_val=X[300:], y_val=y[300:])
    if names is not None:
        kw["feature_names"] = np.array(names, dtype=str)
    np.savez(path, **kw)


@pytest.mark.parametrize("with_names", [True, False])
def test_cd04_worker_reads_names_from_npz(with_names, tmp_path):
    X, y = _nn_data()
    names = [f"name_{i}" for i in range(8)]
    _write_npz(tmp_path / "split.npz", X, y, names if with_names else None)
    cmd = [PY, str(CODE_ROOT / "train_single_trial.py"), "--model-type", "neural_net",
           "--data-path", str(tmp_path / "split.npz"), "--params", json.dumps(NN_PARAMS),
           "--enable-diagnostics", "--diagnostics-level", "full", "--gradient-probe-rows", "32",
           "--normalize-features", "--use-leaky-relu"]
    p = subprocess.run(cmd, cwd=tmp_path, env=SUB_ENV, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-2000:]
    rep = json.loads((tmp_path / "diagnostics_outputs" / "neural_net_diagnostics.json").read_text())
    got = {e["feature"] for e in rep["model_specific"]["feature_gradients"]["top_10"]}
    if with_names:
        assert rep["feature_names_source"] == "npz" and got <= set(names), credit("CD04_NAMES")
        assert list(rep["column_classes"]["by_feature"]) == names, credit("CD04_NAMES")
    else:
        assert rep["feature_names_source"] == "positional", credit("CD04_POSITIONAL")
        assert got <= {f"feature_{i}" for i in range(8)}


@pytest.mark.parametrize("level", ["legacy", "full"])
def test_cd04_mpo_npz_handoff(level, tmp_path):
    import meta_prediction_optimizer_anti_overfit as mpo
    import logging
    o = object.__new__(mpo.AntiOverfitMetaOptimizer)
    o.output_dir = str(tmp_path)
    o.logger = logging.getLogger("t")
    o.feature_schema = {"feature_names": ["a", "b", "c"]}
    o._cli_enable_diagnostics = True
    o._cli_diagnostics_level = level
    X = np.zeros((5, 3), np.float32)
    path = o._export_split_npz(X, np.zeros(5), X, np.zeros(5))
    with np.load(path) as z:
        files = sorted(z.files)
        got = [str(v) for v in z["feature_names"]] if "feature_names" in z.files else None
    if level == "full":
        assert got == ["a", "b", "c"], credit("CD04_NPZ")
    else:
        assert files == ["X_train", "X_val", "y_train", "y_val"]     # unchanged handoff


# ─────────────────────────────────────────────────────────────────────────────
# CD-05 tree reports: trial-worker Booster path
# ─────────────────────────────────────────────────────────────────────────────

FAMILIES = ["xgboost", "lightgbm", "catboost"]


def _train_tree(fam, tmp, tag, enable, opts):
    Xt, yt, Xv, yv, _ = tree_data()
    fn = {"xgboost": tst.train_xgboost, "lightgbm": tst.train_lightgbm, "catboost": tst.train_catboost}[fam]
    ext = {"xgboost": ".json", "lightgbm": ".txt", "catboost": ".cbm"}[fam]
    with force_cpu_trees():
        r = fn(Xt, yt, Xv, yv, dict(TREE_PARAMS), save_path=str(tmp / f"{fam}_{tag}{ext}"),
               enable_diagnostics=enable, diagnostics_options=opts)
    return r, tmp / f"{fam}_{tag}{ext}"


def _tree_predict(fam, path, X):
    if fam == "xgboost":
        import xgboost as xgb
        b = xgb.Booster(); b.load_model(str(path)); return b.predict(xgb.DMatrix(X))
    if fam == "lightgbm":
        import lightgbm as lgb
        return lgb.Booster(model_file=str(path)).predict(X)
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(); m.load_model(str(path)); return m.predict(X)


@pytest.mark.parametrize("fam", FAMILIES)
def test_cd05_worker_tree_report_and_bit_identity(fam, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _, _, Xv, _, names = tree_data()
    opts = {"level": "full", "feature_names": names, "names_source": "npz"}
    r_off, p_off = _train_tree(fam, tmp_path, "off", False, None)
    r_full, p_full = _train_tree(fam, tmp_path, "full", True, opts)
    keys = ("train_mse", "val_mse", "r2", "best_iteration")
    assert [r_off[k] for k in keys] == [r_full[k] for k in keys], credit(f"CD05_BITID_{fam}")
    assert np.array_equal(_tree_predict(fam, p_off, Xv), _tree_predict(fam, p_full, Xv)), credit(f"CD05_BITID_{fam}")
    if fam != "catboost":
        assert sha(p_off) == sha(p_full), credit(f"CD05_BITID_{fam}")
    rep = json.loads((tmp_path / "diagnostics_outputs" / f"{fam}_diagnostics.json").read_text())
    canon = json.loads((tmp_path / "diagnostics_outputs" / "training_diagnostics.json").read_text())
    ts = rep["training_summary"]
    assert rep.get("diagnostics_level") == "full", credit(f"CD05_WORKER_{fam}")
    assert ts["rounds_captured"] == ts["rounds_run"] > 1, credit(f"CD05_WORKER_{fam}")
    assert ts["final_train_loss"] is not None and ts["final_val_loss"] is not None, credit(f"CD05_WORKER_{fam}")
    fi = rep["model_specific"]["feature_importance"]
    assert fi["top_10"][0]["feature"] in names and fi["top_10"][0]["column_class"] == "local", credit(f"CD05_WORKER_{fam}")
    assert rep["column_classes"]["by_feature"]["konst_global"] == "constant"
    assert canon == rep
    if fam == "lightgbm" and r_full["best_iteration"] + 20 <= TREE_PARAMS["n_estimators"]:
        # early stopping fired: lgb.train trimmed the Booster to best_iteration; the 20 rounds
        # after it are known only from the record_evaluation callback
        assert ts["used_rounds"] == r_full["best_iteration"], credit("CD05_LGB_AFTER_BEST")
        assert ts["rounds_run"] == r_full["best_iteration"] + 20, credit("CD05_LGB_AFTER_BEST")
    # graded by the fixed rules without the B1 artefact (ratio is real, not 0/1)
    g = thc.grade_report(rep, {"grading_rules": "fixed"})
    assert g["metrics"]["early_stop_ratio"] is not None and g["metrics"]["best_round_key"] == "best_val_round"


def test_unverified5_boosters_have_no_feature_importances_():
    """Settles plan UNVERIFIED #5 / chapter §17.1 on the installed versions (executed, CPU)."""
    import xgboost as xgb, lightgbm as lgb
    Xt, yt, _, _, _ = tree_data()
    b = xgb.train({"device": "cpu"}, xgb.DMatrix(Xt, label=yt), 5)
    lb = lgb.train({"device": "cpu", "verbosity": -1}, lgb.Dataset(Xt, label=yt), 5)
    assert not hasattr(b, "feature_importances_") and not hasattr(lb, "feature_importances_"), credit("UNVERIFIED5")


@pytest.mark.parametrize("fam", ["xgboost", "catboost"])
def test_worker_tree_legacy_report_identical_to_base(fam, head_dir, tmp_path):
    """Default level: the trial worker's tree report is byte-for-byte today's (minus generated_at)."""
    Xt, yt, Xv, yv, _ = tree_data()
    np.savez(tmp_path / "split.npz", X_train=Xt, y_train=yt, X_val=Xv, y_val=yv)
    driver = tmp_path / "drive.py"
    driver.write_text(_DRIVER)
    out = {}
    for label, root in (("head", head_dir), ("new", CODE_ROOT)):
        wd = tmp_path / label
        wd.mkdir()
        env = dict(SUB_ENV, PYTHONPATH=f"{root}{os.pathsep}{WORKTREE / 'tests'}")
        p = subprocess.run([PY, str(driver), fam, str(tmp_path / "split.npz")], cwd=wd, env=env,
                           capture_output=True, text=True, timeout=300)
        assert p.returncode == 0, p.stderr[-2000:]
        rep = json.loads((wd / "diagnostics_outputs" / f"{fam}_diagnostics.json").read_text())
        rep.pop("generated_at")
        out[label] = rep
    assert out["head"] == out["new"], credit("LEGACY_REPORT_IDENTITY")


_DRIVER = r'''
import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from ch14_cpu_patch import force_cpu_trees, TREE_PARAMS
import train_single_trial as t
fam, npz = sys.argv[1], sys.argv[2]
z = np.load(npz)
fn = {"xgboost": t.train_xgboost, "catboost": t.train_catboost, "lightgbm": t.train_lightgbm}[fam]
with force_cpu_trees():
    fn(z["X_train"], z["y_train"], z["X_val"], z["y_val"], dict(TREE_PARAMS), None, True)
'''


def test_nn_legacy_report_identical_to_base(head_dir, tmp_path):
    driver = tmp_path / "drive_nn.py"
    driver.write_text(r'''
import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, torch
import train_single_trial as t
rng = np.random.default_rng(1); X = rng.normal(size=(400, 8)).astype(np.float32); X[:, 7] = 2.0
y = (X[:, 0] + 0.3 * rng.normal(size=400)).astype(np.float32)
torch.manual_seed(0)
t.train_neural_net(X[:300], y[:300], X[300:], y[300:], {"hidden_layers": [16, 8], "epochs": 12,
    "early_stopping_patience": 50}, None, True, normalize_features=True, use_leaky_relu=True)
''')
    out = {}
    for label, root in (("head", head_dir), ("new", CODE_ROOT)):
        wd = tmp_path / label
        wd.mkdir()
        p = subprocess.run([PY, str(driver)], cwd=wd, env=dict(SUB_ENV, PYTHONPATH=str(root)),
                           capture_output=True, text=True, timeout=300)
        assert p.returncode == 0, p.stderr[-2000:]
        rep = json.loads((wd / "diagnostics_outputs" / "neural_net_diagnostics.json").read_text())
        rep.pop("generated_at")
        out[label] = rep
    assert out["head"] == out["new"], credit("LEGACY_REPORT_IDENTITY")


# ─────────────────────────────────────────────────────────────────────────────
# CD-05 inline final path (mpo MultiModelTrainer, sklearn API)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("fam", FAMILIES)
@pytest.mark.parametrize("level", ["legacy", "full"])
def test_cd05_inline_final_tree_report(fam, level, tmp_path, monkeypatch):
    import logging
    import meta_prediction_optimizer_anti_overfit as mpo
    monkeypatch.chdir(tmp_path)
    Xt, yt, Xv, yv, names = tree_data()
    trainer = mpo.MultiModelTrainer(device="cpu")
    params = {"n_estimators": 60, "max_depth": 3}
    r_ref = trainer.train_model(fam, Xt, yt, Xv, yv, dict(params))
    r = trainer.train_model(fam, Xt, yt, Xv, yv, dict(params))
    o = object.__new__(mpo.AntiOverfitMetaOptimizer)
    o.model_type, o.logger = fam, logging.getLogger("t")
    o.feature_schema = {"feature_names": names}
    o._cli_enable_diagnostics, o._cli_diagnostics_level = True, level
    before = r["model"].predict(Xv)
    o._ch14_emit_inline_tree_report(r, Xt, yt, Xv, yv)
    after = r["model"].predict(Xv)
    assert np.array_equal(before, after) and np.array_equal(before, r_ref["model"].predict(Xv))
    assert r["metrics"] == r_ref["metrics"]
    rep_p = tmp_path / "diagnostics_outputs" / f"{fam}_diagnostics.json"
    if level == "legacy":
        assert not rep_p.exists()                                     # default unchanged
        return
    assert rep_p.exists(), credit("CD05_INLINE")
    rep = json.loads(rep_p.read_text())
    assert rep["feature_names_source"] == "sidecar", credit("CD05_INLINE")
    assert rep["training_summary"]["rounds_run"] > 1, credit("CD05_INLINE")
    assert rep["model_specific"]["feature_importance"]["top_10"][0]["feature"] in names, credit("CD05_INLINE")
    assert (tmp_path / "diagnostics_outputs" / "training_diagnostics.json").exists()


# ─────────────────────────────────────────────────────────────────────────────
# CD-00 offline re-grade
# ─────────────────────────────────────────────────────────────────────────────

def test_cd00_regrade_counts_and_read_only(tmp_path):
    src = tmp_path / "archive"
    hist = src / "history"
    # 2 history archives (recorded critical, the B1 artefact), 3 raw reports, 1 compare summary, 1 other
    write_json(hist / "neural_net_20261101_000000.json",
               {"diagnostics": nn_report(14, 17), "watcher_severity": "critical", "watcher_issues": [],
                "archived_at": "2026-11-01T00:00:00+00:00", "health_check_version": "1.0.0"})
    write_json(hist / "neural_net_20261102_000000.json",
               {"diagnostics": nn_report(1, 17), "watcher_severity": "critical", "watcher_issues": [],
                "archived_at": "2026-11-02T00:00:00+00:00", "health_check_version": "1.0.0"})
    write_json(src / "a_diagnostics.json", nn_report(14, 17, spread=5000.0))
    write_json(src / "b_diagnostics.json", nn_report(10, 17))                  # ratio 0.59 -> ok
    write_json(src / "c_diagnostics.json", nn_report(4, 17))                   # 0.235 -> warning
    write_json(src / "compare_models_summary_S88_20261101_000000.json",
               {"run_id": "S88_20261031_230000", "models": {}, "winner_best_effort": {"model_type": "catboost"}})
    write_json(src / "model_skip_registry.json", {"neural_net": {"consecutive_critical": 2}})
    files = sorted(p for p in src.rglob("*.json"))
    before = {p: (sha(p), p.stat().st_mtime_ns) for p in files}
    out = tmp_path / "out" / "regrade.json"
    res = thc.regrade([str(src)], {}, str(out))
    after = {p: (sha(p), p.stat().st_mtime_ns) for p in files}
    assert before == after and res["read_only_check"]["changed"] == 0, credit("CD00_READONLY")
    assert sorted(p.name for p in src.rglob("*")) == sorted(p.name for p in files + [hist])
    assert res["graded_reports"] == 5, credit("CD00_COUNTS")
    assert res["legacy"] == {"critical": 5}, credit("CD00_COUNTS")
    assert res["fixed"] == {"ok": 2, "warning": 1, "critical": 2}, credit("CD00_COUNTS")
    assert res["recorded_grades"] == 2 and res["recorded_equals_legacy_recomputed"] == 2
    assert res["files_by_kind"] == {"history_archive": 2, "report": 3, "compare_summary": 1, "other": 1}
    assert res["fixed_critical_reasons"] == {"early_stop_ratio": 1, "gradient_spread": 1}
    assert res["time_matched_to_compare_run"] == 1 and res["time_matched_graded_model_not_winner"] == 1
    assert json.loads(out.read_text())["graded_reports"] == 5
    with pytest.raises(FileExistsError):
        thc.regrade([str(src)], {}, str(out))
    with pytest.raises(ValueError):
        thc.regrade([str(src)], {}, str(src / "inside.json"))
    assert not (src / "inside.json").exists()
