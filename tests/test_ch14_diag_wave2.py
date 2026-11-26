#!/usr/bin/env python3
"""
CH14-DIAG wave 2 tests (2026-11-23). CPU only, synthetic fixtures; the one real file read is the
ch14-learn feature manifest, read-only and hash-checked.

Items: CD-07 run identity + stale guard, CD-08 unused_feature_pct, CD-09 skip expiry, CD-10 history
pruning, CD-11 training health log + advisor reader, CD-12 LLM prompt fields, CD-23 dead-unit
metric, TODO-CH14-4 run-level names from the feature manifest, and default-unchanged gates against
the code at HEAD_REV, the main commit this lands on (7f4505a, full sha below;
landing re-fit 2026-11-25, was 41ad509). Everything the suite writes is under --basetemp.

Every credited assertion carries a [CREDIT:<tag>] marker; tests/ch14_wave2_mutants.py applies each
mutant and requires the run to fail on its credited tag with the mutated line reached.

Run (from the worktree root), pinned per the lead's 2026-11-23 19:10 load note:
    taskset -c 4,5 nice -n 15 python3 -m pytest -q tests/test_ch14_diag_wave2.py \
        --basetemp=data/ch14/w2/pytest_tmp/wave2 -p no:cacheprovider
Code under test is imported from $CH14_CODE_ROOT (default: the worktree root).
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # no GPU, set before any ML import
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

WORKTREE = Path(__file__).resolve().parents[1]
CODE_ROOT = Path(os.environ.get("CH14_CODE_ROOT", str(WORKTREE))).resolve()
HEAD_REV = "7f4505aaccebdd284393eb3565e75d09d1c0f3b4"   # main at landing (full sha, never HEAD or a merge base)
# the feature manifest is on main (blob a8f67496, the same as the ch14-learn copy this test first read)
LEARN_MANIFEST = WORKTREE / "models" / "feature_manifest.json"
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(1, str(Path(__file__).resolve().parent))

import training_diagnostics as td          # noqa: E402
import training_health_check as thc        # noqa: E402
import train_single_trial as tst           # noqa: E402
import diagnostics_llm_analyzer as dla     # noqa: E402
import parameter_advisor as pa             # noqa: E402

from ch14_cpu_patch import force_cpu_trees, TREE_PARAMS  # noqa: E402

PY = sys.executable
SUB_ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(CODE_ROOT),
               PYTHONDONTWRITEBYTECODE="1")


def credit(tag):
    return f"[CREDIT:{tag}]"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(obj if isinstance(obj, str) else json.dumps(obj, indent=2))
    return path


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HEAD_FILES = ("training_health_check.py", "training_diagnostics.py", "diagnostics_llm_analyzer.py",
              "agents/contexts/advisor_bundle.py")


@pytest.fixture(scope="session")
def head_dir(tmp_path_factory):
    """Code at HEAD_REV for the default-unchanged gates (git show; never checked out), under --basetemp."""
    HEAD_DIR = tmp_path_factory.mktemp(f"head_w2_{HEAD_REV[:7]}")
    for fn in HEAD_FILES:
        out = subprocess.run(["git", "-C", str(WORKTREE), "show", f"{HEAD_REV}:{fn}"],
                             capture_output=True, check=True).stdout
        p = HEAD_DIR / fn
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(out)
    return HEAD_DIR


def nn_report(best_val_round=14, rounds=17, train=0.90, val=0.95, model_type="neural_net", **extra):
    """Report shaped like a real schema-1.1.0 file (keys as td writes them)."""
    rep = {"status": "complete",
           "training_summary": {"rounds_captured": rounds, "final_train_loss": train,
                                "final_val_loss": val, "best_val_loss": val - 0.01,
                                "best_val_round": best_val_round, "overfit_gap": val - train},
           "diagnosis": {"severity": "ok", "issues": [], "suggested_fixes": [], "confidence": 0.9},
           "model_specific": {"gradient_health": {"vanishing": False, "exploding": False,
                                                  "dead_neuron_pct": 0.0, "gradient_spread_ratio": 1.0},
                              "layer_health": {"network.0": {"dead_pct": 0.0, "gradient_norm": 0.01}}},
           "round_data_truncated": False, "round_data_sample": [],
           "model_type": model_type, "generated_at": "2026-11-23T10:00:00", "schema_version": "1.1.0"}
    rep.update(extra)
    return rep


def policies(cwd, **td_keys):
    write_json(Path(cwd) / "watcher_policies.json", {"training_diagnostics": td_keys})


def registry(cwd):
    p = Path(cwd) / "diagnostics_outputs" / "model_skip_registry.json"
    return json.loads(p.read_text()) if p.exists() else {}


def history(cwd):
    return sorted(p.name for p in (Path(cwd) / "diagnostics_outputs" / "history").glob("*.json"))


# ─────────────────────────────────────────────────────────────────────────────
# CD-07 run identity and stale guard
# ─────────────────────────────────────────────────────────────────────────────

def test_cd07_options_bundle_and_child_flags():
    ns = SimpleNamespace(enable_diagnostics=True, diagnostics_level="full", gradient_probe_rows=64,
                         diagnostics_run_id=None, diagnostics_run_started_at=None,
                         dead_unit_definition="linear_zero", unused_gradient_epsilon=None,
                         feature_manifest=None)
    o = td.diagnostics_options_from_args(ns)
    assert o["run_id"] and o["run_id"].startswith("s5-") and o["run_started_at"], credit("CD07_BUNDLE")
    flags = td.diagnostics_child_flags(o)
    assert flags[flags.index("--diagnostics-run-id") + 1] == o["run_id"], credit("CD07_BUNDLE")
    assert flags[:2] == ["--diagnostics-level", "full"]
    # a child given the parent's id keeps it
    ns2 = SimpleNamespace(**{**vars(ns), "diagnostics_run_id": o["run_id"],
                             "diagnostics_run_started_at": o["run_started_at"]})
    assert td.diagnostics_options_from_args(ns2)["run_id"] == o["run_id"], credit("CD07_BUNDLE")
    # legacy, or diagnostics off: no bundle, no flags
    assert td.diagnostics_options_from_args(SimpleNamespace(**{**vars(ns), "diagnostics_level": "legacy"})) is None
    assert td.diagnostics_options_from_args(SimpleNamespace(**{**vars(ns), "enable_diagnostics": False})) is None
    assert td.diagnostics_child_flags(None) == []


def test_cd07_pointer_and_children_share_run_id(tmp_path, monkeypatch):
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
    opts = {"level": "full", "run_id": "s5-test-R1", "run_started_at": "2026-11-23T20:00:00+00:00"}
    rc = mpo._s88_run_compare_models({"survivors": "s.json", "lottery_data": "l.json", "trials": 1,
                                      "enable_diagnostics": True, "diagnostics_level": "full",
                                      "diagnostics_options": opts})
    assert rc == 0
    ptr = json.loads((Path("diagnostics_outputs") / thc.COMPARE_POINTER_FILENAME).read_text())
    assert ptr["diagnostics_run_id"] == "s5-test-R1", credit("CD07_POINTER")
    assert ptr["diagnostics_run_started_at"] == opts["run_started_at"], credit("CD07_POINTER")
    assert str(ptr["run_id"]).startswith("S88_")                      # S88 compare id untouched
    ids = {c[c.index("--diagnostics-run-id") + 1] for c in calls if "--diagnostics-run-id" in c}
    assert len(calls) == 4 and ids == {"s5-test-R1"}, credit("CD07_POINTER")


def _compare_dir(cwd, pointer_id="R1", winner_id="R1", nn_id="R1"):
    d = Path(cwd) / "diagnostics_outputs"
    write_json(d / "training_diagnostics.json", nn_report(1, 17, diagnostics_run_id=nn_id))
    write_json(d / "neural_net_diagnostics.json", nn_report(1, 17, diagnostics_run_id=nn_id))
    write_json(d / "catboost_diagnostics.json",
               nn_report(14, 17, model_type="catboost", diagnostics_run_id=winner_id))
    write_json(d / thc.COMPARE_POINTER_FILENAME, {
        "run_id": "S88_20261123_120000", "diagnostics_run_id": pointer_id,
        "models": {"neural_net": {}, "catboost": {}},
        "winner_best_effort": {"model_type": "catboost", "score": 0.4}})
    return d


def test_cd07_fixed_path_stale_winner_not_graded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    policies(tmp_path, grading_rules="fixed")
    _compare_dir(tmp_path, pointer_id="R1", winner_id="R0")
    r = thc.check_training_health()
    assert r["severity"] == "stale" and r["action"] == "PROCEED_WITH_NOTE", credit("CD07_STALE")
    assert r["freshness"]["report_run_id"] == "R0" and r["freshness"]["pointer_run_id"] == "R1"
    assert registry(tmp_path) == {} and history(tmp_path) == [], credit("CD07_STALE")
    # same run: graded as before (catboost healthy, NN critical -> warning note)
    _compare_dir(tmp_path, pointer_id="R1", winner_id="R1")
    r2 = thc.check_training_health()
    assert r2["severity"] == "warning" and r2["freshness"]["freshness"] == "fresh", credit("CD07_FRESH")
    assert r2["per_family"]["neural_net"]["severity"] == "critical"


def test_cd07_stale_family_listed_not_graded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    policies(tmp_path, grading_rules="fixed", grade_scope="all_families")
    _compare_dir(tmp_path, pointer_id="R1", winner_id="R1", nn_id="R0")   # NN report from run R0
    r = thc.check_training_health()
    assert r["per_family"]["neural_net"]["severity"] == "stale", credit("CD07_FAMILY")
    assert r["severity"] == "ok" and r["model_type"] == "catboost", credit("CD07_FAMILY")


@pytest.mark.parametrize("action", ["RETRY", "grade"])
def test_cd07_stale_action_setting(action, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    policies(tmp_path, grading_rules="fixed", stale_report_action=action)
    _compare_dir(tmp_path, pointer_id="R1", winner_id="R0")
    r = thc.check_training_health()
    if action == "RETRY":
        assert r["action"] == "RETRY" and r["severity"] == "stale" and registry(tmp_path) == {}, \
            credit("CD07_STALE_ACTION")
    else:  # graded anyway (catboost ok, NN critical -> warning), with the freshness note
        assert r["severity"] == "warning" and any("freshness stale" in i for i in r["issues"]), \
            credit("CD07_STALE_ACTION")


def test_cd07_legacy_path_only_with_caller_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    policies(tmp_path)
    write_json(tmp_path / "diagnostics_outputs" / "training_diagnostics.json",
               nn_report(1, 17, diagnostics_run_id="R0"))
    r = thc.check_training_health(expected_run_id="R9")
    assert r["severity"] == "stale" and registry(tmp_path) == {}, credit("CD07_LEGACY_CALLER")
    r2 = thc.check_training_health(expected_run_id="R0")
    assert r2["severity"] == "critical" and r2["freshness"]["freshness"] == "fresh"
    # no stamp in the report: the file time decides
    write_json(tmp_path / "diagnostics_outputs" / "training_diagnostics.json", nn_report(1, 17))
    later = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    r3 = thc.check_training_health(run_started_at=later)
    assert r3["severity"] == "stale" and "before run start" in r3["freshness"]["reason"], credit("CD07_MTIME")
    earlier = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    assert thc.check_training_health(run_started_at=earlier)["freshness"]["freshness"] == "fresh"


def test_cd07_unknown_setting_value_fails_closed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    policies(tmp_path, stale_report_action="IGNORE")
    write_json(tmp_path / "diagnostics_outputs" / "training_diagnostics.json", nn_report(14, 17))
    r = thc.check_training_health()
    assert r["model_type"] == "unknown" and "stale_report_action" in r["issues"][0], credit("CD07_FAIL_CLOSED")


# ─────────────────────────────────────────────────────────────────────────────
# CD-08 unused_feature_pct
# ─────────────────────────────────────────────────────────────────────────────

def test_cd08_unused_metric_and_classes():
    names = [f"f{i:02d}" for i in range(89)]
    vec = np.ones(89)
    vec[:50] = 0.0                                                     # 50 of 89 unused
    by = {n: ("constant" if i < 36 else "local") for i, n in enumerate(names)}
    u = td.unused_features(vec, "gain", names, by)
    assert u["unused"] == 50 and abs(u["pct"] - 100 * 50 / 89) < 1e-9, credit("CD08_METRIC")
    assert u["by_class"]["constant"] == {"columns": 36, "unused": 36, "pct": 100.0}, credit("CD08_METRIC")
    assert abs(u["pct_excluding_constant"] - 100 * 14 / 53) < 1e-9, credit("CD08_METRIC")
    g = td.unused_features(np.array([1e-9, 1e-3, 0.0]), "input_gradient_mean_abs", ["a", "b", "c"], {})
    assert g["unused_names"] == ["a", "c"] and "1e-08" in g["rule"]


@pytest.mark.parametrize("mode,pct,want", [("off", 56.2, "ok"), ("all", 56.2, "warning"),
                                           ("all", 75.0, "critical"), ("exclude_constant", 56.2, "ok")])
def test_cd08_fixed_grading(mode, pct, want):
    rep = nn_report(14, 17, unused_feature_pct=pct,
                    unused_features={"pct": pct, "pct_excluding_constant": 26.4})
    g = thc.grade_report(rep, {"grading_rules": "fixed", "unused_feature_grading": mode})
    assert g["severity"] == want, credit("CD08_GRADE")


def test_cd08_tree_report_carries_unused(tmp_path):
    import xgboost as xgb
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 6)).astype(np.float32)
    X[:, 5] = 4.0                                                      # constant: never split
    y = (X[:, 0] + 0.1 * rng.normal(size=300)).astype(np.float32)
    m = xgb.XGBRegressor(n_estimators=20, max_depth=2, device="cpu", n_jobs=2).fit(X[:200], y[:200])
    names = ["a", "b", "c", "d", "e", "konst"]
    d = td.build_tree_diagnostics_full(m, "xgboost", X[:200], y[:200], X[200:], y[200:],
                                       feature_names=names, names_source="caller",
                                       options={"run_id": "R7", "run_started_at": "t0"})
    rep = d.get_report()
    uf = rep.get("unused_features")
    assert uf and "konst" in uf["unused_names"] and uf["by_class"]["constant"]["unused"] == 1, credit("CD08_REPORT")
    assert rep["unused_feature_pct"] == uf["pct"]
    assert rep["diagnostics_run_id"] == "R7" and rep["diagnostics_run_started_at"] == "t0", credit("CD07_STAMP")


# ─────────────────────────────────────────────────────────────────────────────
# CD-09 skip expiry
# ─────────────────────────────────────────────────────────────────────────────

T0 = datetime(2026, 11, 23, 0, 0, tzinfo=timezone.utc)
ENTRY = {"consecutive_critical": 3, "last_critical": T0.isoformat()}


def test_cd09_expiry_clock():
    # [BENCHING 2026-11-26] expiry applies with benching on; the default is now off (never skipped)
    hours = {"max_consecutive_critical": 3, "skip_duration_hours": 24, "expiry_mode": "hours", "benching": "on"}
    assert thc.skip_active(ENTRY, hours, now=T0 + timedelta(hours=23))[0] is True, credit("CD09_EXPIRY")
    assert thc.skip_active(ENTRY, hours, now=T0 + timedelta(hours=25))[0] is False, credit("CD09_EXPIRY")
    reset = {"max_consecutive_critical": 3, "skip_duration_hours": 24, "benching": "on"}
    for h in (25, 1000):
        assert thc.skip_active(ENTRY, reset, now=T0 + timedelta(hours=h))[0] is True, credit("CD09_RESET")
    assert thc.skip_active({"consecutive_critical": 2}, hours)[0] is False
    assert thc.skip_active({"consecutive_critical": 3}, hours)[0] is True       # no time: kept
    with pytest.raises(ValueError):
        thc.skip_active(ENTRY, {"expiry_mode": "days"})


@pytest.mark.parametrize("mode", ["until_reset", "hours"])
def test_cd09_both_readers(mode, tmp_path, monkeypatch):
    import meta_prediction_optimizer_anti_overfit as mpo
    monkeypatch.chdir(tmp_path)
    old = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
    write_json(tmp_path / "diagnostics_outputs" / "model_skip_registry.json",
               {"neural_net": {"consecutive_critical": 3, "last_critical": old}})
    rules = {"max_consecutive_critical": 3, "skip_duration_hours": 24, "benching": "on"}  # [BENCHING]
    if mode == "hours":
        rules["expiry_mode"] = "hours"
    write_json(tmp_path / "watcher_policies.json", {"training_diagnostics": {"model_skip_rules": rules}})
    assert thc.is_model_skipped("neural_net") is (mode == "until_reset"), credit("CD09_THC")
    calls = []

    def fake_run(cmd, env=None, stdout=None, stderr=None, **kw):
        calls.append(cmd[cmd.index("--model-type") + 1])
        write_json(Path("models/reinforcement/best_model.meta.json"), {"training_metrics": {"r2": 0.1}})
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    mpo._s88_run_compare_models({"survivors": "s.json", "lottery_data": "l.json", "trials": 1,
                                 "enable_diagnostics": False})
    assert ("neural_net" in calls) is (mode == "hours"), credit("CD09_MPO")


def test_cd09_skip_status_default_identical(head_dir, tmp_path, monkeypatch):
    head = load_module("thc_head_w2", head_dir / "training_health_check.py")
    monkeypatch.chdir(tmp_path)
    write_json(tmp_path / "diagnostics_outputs" / "model_skip_registry.json",
               {"neural_net": dict(ENTRY), "catboost": {"consecutive_critical": 1, "last_critical": None}})
    # [BENCHING 2026-11-26] the default is now off; benching on is the setting identical to HEAD_REV
    write_json(tmp_path / "watcher_policies.json",
               {"training_diagnostics": {"model_skip_rules": {"benching": "on"}}})
    assert thc.get_skip_status() == head.get_skip_status(), credit("CD09_DEFAULT")
    assert thc.is_model_skipped("neural_net") == head.is_model_skipped("neural_net")


# ─────────────────────────────────────────────────────────────────────────────
# CD-10 history pruning
# ─────────────────────────────────────────────────────────────────────────────

def _fill_history(cwd, n=105):
    h = Path(cwd) / "diagnostics_outputs" / "history"
    h.mkdir(parents=True, exist_ok=True)
    base = time.time() - 10_000
    for i in range(n):
        p = h / f"neural_net_20260101_{i:06d}.json"
        p.write_text("{}")
        os.utime(p, (base + i, base + i))
    for i in range(3):
        p = h / f"root_cause_2026010{i}.json"
        p.write_text("{}")
        os.utime(p, (base - 100 + i, base - 100 + i))                  # older than everything
    for i in range(2):
        (h / f"compare_models_2026010{i}.json").write_text("{}")
    return h


@pytest.mark.parametrize("setting", ["off", "move", "delete", "move_with_root_cause"])
def test_cd10_prune(setting, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    keys = {} if setting == "off" else {"history_prune": "on",
                                        "history_prune_mode": "delete" if setting == "delete" else "move"}
    if setting == "move_with_root_cause":
        keys["history_prune_include_root_cause"] = True
    policies(tmp_path, **keys)
    h = _fill_history(tmp_path)
    thc._archive_diagnostics(nn_report(), "ok", [], "neural_net")         # 106 archive files now
    names = [p.name for p in h.glob("*.json")]
    nn = [n for n in names if n.startswith("neural_net_")]
    rc = [n for n in names if n.startswith("root_cause_")]
    cm = [n for n in names if n.startswith("compare_models_")]
    arch = sorted(p.name for p in (tmp_path / "diagnostics_outputs" / "history_archive").glob("*.json"))
    assert len(cm) == 2                                                   # td's own FIFO set untouched
    if setting == "off":
        assert len(nn) == 106 and len(rc) == 3 and arch == []
    elif setting == "move_with_root_cause":
        assert len(nn) + len(rc) == 100 and rc == [] and len(arch) == 9, credit("CD10_ROOT_CAUSE")
    else:
        assert len(nn) == 100 and len(rc) == 3, credit("CD10_PRUNE")
        assert "neural_net_20260101_000000.json" not in names, credit("CD10_PRUNE")  # oldest went
        assert len(arch) == (6 if setting == "move" else 0), credit("CD10_MODE")


def test_cd10_td_fifo_defaults_unchanged(head_dir, tmp_path):
    head = load_module("td_head_w2", head_dir / "training_diagnostics.py")
    for mod, sub in ((head, "a"), (td, "b")):
        h = tmp_path / sub
        h.mkdir()
        for i in range(7):
            p = h / f"compare_models_{i}.json"
            p.write_text("{}")
            os.utime(p, (1000 + i, 1000 + i))
        (h / "neural_net_x.json").write_text("{}")
        mod._prune_history_fifo(str(h), max_files=4)
    assert sorted(os.listdir(tmp_path / "a")) == sorted(os.listdir(tmp_path / "b")), credit("CD10_DEFAULT")
    # the default prune deletes in place: nothing is moved or written anywhere else (landing re-fit)
    assert sorted(os.listdir(tmp_path)) == ["a", "b"], credit("CD10_DEFAULT")


# ─────────────────────────────────────────────────────────────────────────────
# CD-11 training health log and advisor reader
# ─────────────────────────────────────────────────────────────────────────────

def test_cd11_log_writer_and_reader(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log = tmp_path / "diagnostics_outputs" / thc.TRAINING_HEALTH_LOG_FILENAME
    write_json(tmp_path / "diagnostics_outputs" / "training_diagnostics.json",
               nn_report(1, 17, diagnostics_run_id="R3"))
    policies(tmp_path)
    thc.check_training_health()
    assert not log.exists()                                               # default off
    os.remove(tmp_path / "diagnostics_outputs" / "model_skip_registry.json")
    policies(tmp_path, training_health_log="on")
    thc.check_training_health(expected_run_id="R3")                       # critical -> RETRY
    thc.check_training_health(expected_run_id="R3")                       # RETRY again
    os.remove(tmp_path / "diagnostics_outputs" / "training_diagnostics.json")
    thc.check_training_health()                                           # absent -> PROCEED
    rows = [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
    assert [r["action"] for r in rows] == ["RETRY", "RETRY", "PROCEED"], credit("CD11_LOG")
    assert rows[1]["registry_consecutive_critical"] == 2 and rows[0]["report_run_id"] == "R3", credit("CD11_LOG")
    assert {"severity", "model_type", "expected_run_id", "schema"} <= set(rows[0])
    st = rows[0].get("settings") or {}
    assert st.get("training_health_log") == "on" and st.get("grading_rules") == "legacy", credit("RUN_RECORD_LOG")
    got = pa.DiagnosticsLoader(str(tmp_path)).load_training_health(2)
    assert [g["action"] for g in got] == ["PROCEED", "RETRY"], credit("CD11_READER")   # newest first


def test_cd11_advisor_gate(tmp_path, head_dir):
    write_json(tmp_path / "diagnostics_outputs" / thc.TRAINING_HEALTH_LOG_FILENAME,
               json.dumps({"model_type": "neural_net", "severity": "critical", "action": "RETRY"}))
    adv = pa.StrategyAdvisor(state_dir=str(tmp_path))
    sig = {"hit_at_20": 0.1}
    adv._add_training_health(sig, {"training_diagnostics": {}})
    assert sig == {"hit_at_20": 0.1}, credit("CD11_DEFAULT_OFF")
    adv._add_training_health(sig, {"training_diagnostics": {"advisor_reads_training_health": True}})
    assert sig["training_health"]["by_model"] == {"neural_net": {"critical": 1}}, credit("CD11_GATE")
    with pytest.raises(ValueError):
        adv._add_training_health({}, {"training_diagnostics": {"advisor_reads_training_health": "yes"}})
    # bundle: without the key, byte-identical to HEAD_REV; with it, one added line
    from agents.contexts import advisor_bundle as ab
    head_ab = load_module("ab_head_w2", head_dir / "agents" / "contexts" / "advisor_bundle.py")
    kw = dict(diagnostics=[{}], telemetry=[], policy_history=[], metrics={"pcs": 0.1},
              signals={"hit_at_20": 0.1})
    assert ab.build_advisor_bundle(**kw) == head_ab.build_advisor_bundle(**kw), credit("CD11_DEFAULT_OFF")
    kw["signals"] = dict(sig)
    assert "Step 5 Training Health" in ab.build_advisor_bundle(**kw), credit("CD11_GATE")


# ─────────────────────────────────────────────────────────────────────────────
# CD-12 LLM prompt from the real report fields (no LLM is called)
# ─────────────────────────────────────────────────────────────────────────────

def _full_report():
    top = [{"feature": f"feat_{i}", "importance": 10.0 - i, "column_class": "local" if i % 2 else "run"}
           for i in range(10)]
    rep = nn_report(3, 18, train=0.5, val=0.8, model_type="catboost", diagnostics_run_id="R5",
                    unused_feature_pct=40.4, column_classes={"counts": {"constant": 36, "run": 14}},
                    round_index_base=0)
    rep["model_specific"] = {"feature_importance": {"top_10": top}}
    return rep


def test_cd12_prompt_report_fields(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    d = tmp_path / "diagnostics_outputs"
    write_json(d / "training_diagnostics.json", _full_report())
    write_json(d / "catboost_diagnostics.json", _full_report())
    write_json(d / thc.COMPARE_POINTER_FILENAME, {"run_id": "S88_x", "diagnostics_run_id": "R5",
                                                  "models": {"catboost": {}, "neural_net": {}},
                                                  "winner_best_effort": {"model_type": "catboost"}})

    def no_llm(*a, **k):
        raise AssertionError("an LLM was called")
    monkeypatch.setattr(dla, "request_llm_diagnostics_analysis", no_llm)
    p = dla.build_diagnostics_prompt(str(d / "training_diagnostics.json"), prompt_fields="report")
    block = json.loads(p.split("=== STEP OUTPUT METRICS ===\n", 1)[1].split("\n\n", 1)[0])
    assert block.get("best_val_round") == 3 and block.get("rounds_captured") == 18, credit("CD12_PROMPT")
    assert abs((block.get("overfit_gap") or 0) - 0.3) < 1e-9, credit("CD12_PROMPT")
    named = [e for e in block.get("top_features", []) if str(e.get("feature", "")).startswith("feat_")
             and e.get("column_class")]
    assert len(named) >= 5, credit("CD12_PROMPT")
    fams = block.get("compare_models_families") or {}
    assert fams.get("winner") == "catboost" and \
        fams.get("families", {}).get("neural_net", {}).get("severity") == "absent", credit("CD12_FAMILIES")
    legacy = dla.build_diagnostics_prompt(str(d / "training_diagnostics.json"), prompt_fields="legacy")
    lb = json.loads(legacy.split("=== STEP OUTPUT METRICS ===\n", 1)[1].split("\n\n", 1)[0])
    assert lb["training_rounds_best"] == 0 and lb["top_5_features"] == []   # what the legacy keys see


def test_cd12_old_archive_fallback():
    old = {"model_type": "neural_net", "diagnosis": {"severity": "warning"},
           "training_rounds": {"total": 40, "best": 9},
           "feature_importance": {"values": {"x1": 3, "x2": 2}},
           "nn_specific": {"feature_gradient_spread": 12.0, "layer_health": {"l0": {"dead_pct": 1.0}}}}
    s = dla.report_fields_summary(old)
    assert s.get("rounds_captured") == 40 and s.get("best_val_round") == 9, credit("CD12_FALLBACK")
    assert [e["feature"] for e in s["top_features"]] == ["x1", "x2"] and s["nn_gradient_spread"] == 12.0


def test_cd12_default_prompt_identical_to_base(head_dir, tmp_path, monkeypatch):
    head = load_module("dla_head_w2", head_dir / "diagnostics_llm_analyzer.py")
    monkeypatch.chdir(tmp_path)
    p = write_json(tmp_path / "diagnostics_outputs" / "training_diagnostics.json", _full_report())
    assert dla.build_diagnostics_prompt(str(p)) == head.build_diagnostics_prompt(str(p)), credit("CD12_DEFAULT")
    policies(tmp_path, llm_prompt_fields="report")
    assert dla.build_diagnostics_prompt(str(p)) != head.build_diagnostics_prompt(str(p))


# ─────────────────────────────────────────────────────────────────────────────
# CD-23 dead-unit metric that can fire
# ─────────────────────────────────────────────────────────────────────────────

def _forced_net(dead=True):
    import torch
    import torch.nn as nn
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(4, 3), nn.LeakyReLU(0.01), nn.Linear(3, 1))
    if dead:
        with torch.no_grad():
            net[0].weight[0].zero_()
            net[0].bias[0] = -5.0                          # unit 0: pre-activation -5 on every row
    return net


@pytest.mark.parametrize("definition,dead,want", [
    ("always_negative_preact", True, 100.0 / 3), ("always_negative_preact", False, 0.0),
    ("linear_zero", True, 0.0)])
def test_cd23_dead_unit_metric(definition, dead, want):
    import torch
    net = _forced_net(dead)
    X = torch.randn(64, 4)
    y = torch.randn(64, 1)
    d = td.NNDiagnostics(diagnostics_level="full", capture_every_n=1, dead_unit_definition=definition)
    d.attach(net)
    d.set_gradient_probe(X, y, torch.nn.MSELoss())
    loss = torch.nn.functional.mse_loss(net(X), y)
    loss.backward()
    d.on_round_end(0, float(loss), float(loss))
    ms = d.get_report()["model_specific"]
    got = ms["layer_health"]["0"]["dead_pct"]
    assert abs(got - want) < 1e-6, credit("CD23_FIRES")
    if definition == "always_negative_preact":
        assert ms["gradient_health"]["dead_unit_definition_effective"] == "always_negative_preact", credit("CD23_FIRES")
        assert ms["gradient_health"]["dead_unit_layers_without_activation"] == ["2"]
        assert "dead_pct" not in ms["layer_health"].get("2", {})          # output layer: no activation
    else:
        assert "dead_unit_definition" not in ms["gradient_health"]          # default: report unchanged


def test_cd23_without_probe_says_old_definition_used():
    import torch
    net = _forced_net(True)
    X = torch.randn(16, 4)
    d = td.NNDiagnostics(capture_every_n=1, dead_unit_definition="always_negative_preact")
    d.attach(net)
    net(X).sum().backward()
    d.on_round_end(0, 1.0, 1.0)
    gh = d.get_report()["model_specific"]["gradient_health"]
    assert gh["dead_unit_definition"] == "always_negative_preact"
    assert gh["dead_unit_definition_effective"] == "linear_zero", credit("CD23_EFFECTIVE")


NN_PARAMS = {"hidden_layers": [16, 8], "epochs": 12, "early_stopping_patience": 50, "batch_size": 64}


def _nn_data():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 8)).astype(np.float32)
    X[:, 7] = 2.0
    y = (X[:, 0] + 0.3 * rng.normal(size=400)).astype(np.float32)
    return X, y


def _train_nn(tmp, tag, opts, mode="full"):
    import torch
    X, y = _nn_data()
    torch.manual_seed(0)
    r = tst.train_neural_net(X[:300], y[:300], X[300:], y[300:], dict(NN_PARAMS),
                             save_path=str(tmp / f"{tag}.pth"), enable_diagnostics=opts is not False,
                             normalize_features=True, use_leaky_relu=True, batch_mode=mode,
                             diagnostics_options=opts or None)
    sd = torch.load(str(tmp / f"{tag}.pth"), weights_only=False)["state_dict"]
    rep = tmp / "diagnostics_outputs" / "neural_net_diagnostics.json"
    return r, sd, (json.loads(rep.read_text()) if rep.exists() and opts else None)


@pytest.mark.parametrize("mode", ["full", "mini"])
def test_cd23_cd07_nn_bit_identical(mode, tmp_path, monkeypatch):
    import torch
    monkeypatch.chdir(tmp_path)
    names = [f"n{i}" for i in range(8)]
    base = {"level": "full", "feature_names": names, "names_source": "npz", "gradient_probe_rows": 64,
            "run_id": "R11", "run_started_at": "2026-11-23T20:00:00+00:00"}
    r_off, sd_off, _ = _train_nn(tmp_path, "off", False, mode)
    r_old, sd_old, rep_old = _train_nn(tmp_path, "old", dict(base, dead_unit_definition="linear_zero"), mode)
    r_new, sd_new, rep_new = _train_nn(tmp_path, "new", dict(base, dead_unit_definition="always_negative_preact"), mode)
    assert all(torch.equal(sd_off[k], sd_new[k]) and torch.equal(sd_off[k], sd_old[k]) for k in sd_off), credit("CD23_BITID")
    keys = ("train_mse", "val_mse", "r2", "best_iteration")
    assert [r_off[k] for k in keys] == [r_new[k] for k in keys] == [r_old[k] for k in keys], credit("CD23_BITID")
    assert rep_new["model_specific"]["gradient_health"]["dead_unit_definition_effective"] == "always_negative_preact"
    assert rep_new["diagnostics_run_id"] == "R11", credit("CD07_STAMP")
    assert rep_new.get("diagnostics_settings", {}).get("dead_unit_definition") == "always_negative_preact" and \
        rep_new["diagnostics_settings"]["gradient_probe_rows"] == 64, credit("RUN_RECORD_REPORT")
    assert "unused_features" in rep_new and rep_new["unused_features"]["attribution_kind"] == "input_gradient_mean_abs"


def test_cd07_worker_cli_stamps_run_id(tmp_path):
    X, y = _nn_data()
    np.savez(tmp_path / "split.npz", X_train=X[:300], y_train=y[:300], X_val=X[300:], y_val=y[300:],
             feature_names=np.array([f"name_{i}" for i in range(8)], dtype=str))
    cmd = [PY, str(CODE_ROOT / "train_single_trial.py"), "--model-type", "neural_net",
           "--data-path", str(tmp_path / "split.npz"), "--params", json.dumps(NN_PARAMS),
           "--enable-diagnostics", "--diagnostics-level", "full", "--gradient-probe-rows", "32",
           "--normalize-features", "--use-leaky-relu", "--diagnostics-run-id", "R21",
           "--diagnostics-run-started-at", "2026-11-23T21:00:00+00:00",
           "--dead-unit-definition", "always_negative_preact", "--unused-gradient-epsilon", "1e-6",
           "--feature-manifest", str(tmp_path / "no_such_manifest.json")]
    p = subprocess.run(cmd, cwd=tmp_path, env=SUB_ENV, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-2000:]
    rep = json.loads((tmp_path / "diagnostics_outputs" / "neural_net_diagnostics.json").read_text())
    assert rep["diagnostics_run_id"] == "R21", credit("CD07_WORKER")
    assert "1e-06" in rep["unused_features"]["rule"], credit("CD07_WORKER")
    assert rep["column_classes"]["run_level_names_source"]["status"] == "missing"


# ─────────────────────────────────────────────────────────────────────────────
# TODO-CH14-4 run-level names from the feature manifest
# ─────────────────────────────────────────────────────────────────────────────

def _manifest(path, run_groups=("global",)):
    return write_json(path, {"format": td.FEATURE_MANIFEST_FORMAT,
                             "groups": {"global": ["g_a", "g_b"], "battery": ["b_x"]},
                             "aliases": {"run_context": list(run_groups)},
                             "versions": {}})


def test_ch14_4_run_class_from_manifest(tmp_path):
    m = _manifest(tmp_path / "fm.json")
    X = np.column_stack([np.full(20, 1.0), np.full(20, 2.0), np.full(20, 3.0), np.arange(20.0)])
    names = ["g_a", "konst", "b_x", "loc"]
    cc = td.classify_from_options(X, names, {"feature_manifest": str(m)})
    assert cc["by_feature"]["g_a"] == "run" and cc["by_feature"]["konst"] == "constant", credit("CH14_4_RUN")
    assert cc["by_feature"]["b_x"] == "constant" and cc["by_feature"]["loc"] == "local"
    src = cc["run_level_names_source"]
    assert src["status"] == "ok" and src["groups"] == ["global"] and src["sha256"] == sha(m), credit("CH14_4_RUN")
    assert cc["run_level_names_absent_from_data"] == ["g_b"]
    assert cc["trial_class"] == "unavailable" and "no trial id" in cc["trial_class_reason"]
    missing = td.classify_from_options(X, names, {"feature_manifest": str(tmp_path / "none.json")})
    assert missing["by_feature"]["g_a"] == "constant" and missing["run_level_names_source"]["status"] == "missing"
    write_json(tmp_path / "bad.json", {"format": "other/9", "groups": {}})
    assert td.load_run_level_names(str(tmp_path / "bad.json"))[1]["status"] == "invalid"


def test_ch14_4_real_learn_manifest_read_only():
    assert LEARN_MANIFEST.is_file(), f"{LEARN_MANIFEST} is tracked on main and must exist"
    before = sha(LEARN_MANIFEST)
    names, src = td.load_run_level_names(str(LEARN_MANIFEST))
    assert sha(LEARN_MANIFEST) == before                                  # read only
    man = json.loads(LEARN_MANIFEST.read_text())
    want = sorted({n for g in man["aliases"]["run_context"] for n in man["groups"][g]})
    assert names == want and src["status"] == "ok" and all(n.startswith("global_") for n in names), credit("CH14_4_REAL")
