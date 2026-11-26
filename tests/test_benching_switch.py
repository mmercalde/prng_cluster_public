#!/usr/bin/env python3
"""
Benching switch tests (2026-11-26; owner ruling 2026-11-24 10:47-10:50, confirmed 2026-11-26 07:05-07:07).
CPU only, synthetic fixtures. Everything the suite writes is under --basetemp.

Switch: watcher_policies.json -> training_diagnostics.model_skip_rules.benching, off (default) | on.
Readers: training_health_check.py (check_training_health, is_model_skipped, get_skip_status) and
meta_prediction_optimizer_anti_overfit.py (_s88_run_compare_models, the compare's family list).

  (a) benching off: N criticals, the family is never skipped in either reader; the why-report flag is raised
      once the limit is reached (result key, warning log, one line in the flag file under the temp dir)
  (b) benching on + until_reset equals the code at BASE_REV (git show, never a restated copy)
  (c) benching on + hours: after a timed skip expires the count restarts (full retries again)
  (d) a bad switch value fails loudly
  (e) nothing is written outside the temp dir (repo root, data/ outside --basetemp, catboost_info,
      diagnostics_outputs, git status --ignored)
Every credited assertion carries a [CREDIT:<tag>]; tests/bench_switch_mutants.py turns each tag red.

Run (from the worktree root), one suite at a time:
    PYTHONDONTWRITEBYTECODE=1 nice -n 15 python3 -B -m pytest -q tests/test_benching_switch.py \
        --basetemp=data/bench_switch/pt -p no:cacheprovider
Code under test is imported from $BENCH_CODE_ROOT (default: the worktree root).
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # no GPU, set before any ML import
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import importlib.util
import json
import logging
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

WORKTREE = Path(__file__).resolve().parents[1]
CODE_ROOT = Path(os.environ.get("BENCH_CODE_ROOT", str(WORKTREE))).resolve()
BASE_SHORT = "7f28109"   # main before this change (rebased from 933a4a0; none of the files differ); resolved to its full sha below (never HEAD)
sys.path.insert(0, str(CODE_ROOT))
if CODE_ROOT != WORKTREE:
    sys.path.insert(1, str(WORKTREE))

import training_health_check as thc        # noqa: E402

BASE_FILES = ("training_health_check.py", "meta_prediction_optimizer_anti_overfit.py")
FAMILIES = ["neural_net", "lightgbm", "xgboost", "catboost"]


def credit(tag):
    return f"[CREDIT:{tag}]"


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
    return path


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _full_sha(rev):
    return subprocess.run(["git", "-C", str(WORKTREE), "rev-parse", "--verify", f"{rev}^{{commit}}"],
                          capture_output=True, text=True, check=True).stdout.strip()


BASE_REV = _full_sha(BASE_SHORT)


# ─────────────────────────────────────────────────────────────────────────────
# (e) isolation: snapshot of everything outside --basetemp that a test could write
# ─────────────────────────────────────────────────────────────────────────────

def _tree_state(root, skip=()):
    """{relative path: (size, mtime_ns)} for every file under root, skipping any path under `skip`."""
    root = Path(root)
    out = {}
    if not root.exists():
        return out
    skip_set = {Path(s).resolve() for s in skip}
    for dirpath, dirnames, filenames in os.walk(root):
        dp = Path(dirpath)
        dirnames[:] = [d for d in dirnames if d != ".git" and (dp / d).resolve() not in skip_set]
        for fn in filenames:
            p = dp / fn
            try:
                st = p.lstat()
            except OSError:
                continue
            out[str(p.relative_to(root))] = (st.st_size, st.st_mtime_ns)
    return out


def _snapshot(repo, basetemp):
    repo = Path(repo)
    top = {}
    for p in repo.iterdir():
        if p.name == ".git":
            continue
        st = p.lstat()
        top[p.name] = (st.st_size if p.is_file() else -1, st.st_mtime_ns if p.is_file() else 0)
    watched = {name: _tree_state(repo / name, skip=(basetemp,) if basetemp else ())
               for name in ("data", "catboost_info", "diagnostics_outputs", "models")}
    gp = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--ignored", "--untracked-files=all"],
                        capture_output=True, text=True)
    # [R1 F6] Beta Q8: git's exit status is checked, so a failed git call cannot pass as "no change"
    assert gp.returncode == 0, f"git status failed in {repo} (rc={gp.returncode}): {gp.stderr[-500:]}"
    gs = gp.stdout.splitlines()
    bt = str(Path(basetemp).resolve().relative_to(repo.resolve())) if basetemp and \
        Path(basetemp).resolve().is_relative_to(repo.resolve()) else None
    gs = sorted(l for l in gs if not (bt and l[3:].startswith(bt)) and "__pycache__" not in l)
    return {"top": top, "watched": watched, "git": gs}


_BEFORE = {}


@pytest.fixture(scope="session", autouse=True)
def _isolation_snapshot(tmp_path_factory):
    bt = tmp_path_factory.getbasetemp()
    _BEFORE["basetemp"] = bt
    _BEFORE["snap"] = _snapshot(WORKTREE, bt)
    yield


# ─────────────────────────────────────────────────────────────────────────────
# fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def base_dir(tmp_path_factory):
    """The two reader files at BASE_REV (git show; never checked out), under --basetemp."""
    d = tmp_path_factory.mktemp(f"base_{BASE_SHORT}")
    for fn in BASE_FILES:
        out = subprocess.run(["git", "-C", str(WORKTREE), "show", f"{BASE_REV}:{fn}"],
                             capture_output=True, check=True).stdout
        (d / fn).write_bytes(out)
    return d


def critical_report(model_type="neural_net"):
    return {"model_type": model_type, "status": "complete",
            "training_summary": {"rounds_captured": 17, "final_train_loss": 0.9, "final_val_loss": 0.95,
                                 "best_val_round": 14},
            "diagnosis": {"severity": "critical", "issues": ["synthetic critical"], "suggested_fixes": []}}


def multi_report(winner="catboost"):
    return {"models": {"neural_net": critical_report("neural_net"), winner: critical_report(winner)},
            "comparison": {"winner": winner}}


def policies(cwd, skip_rules=None, **td_keys):
    td = dict(td_keys)
    if skip_rules is not None:
        td["model_skip_rules"] = skip_rules
    write_json(Path(cwd) / "watcher_policies.json", {"training_diagnostics": td})


def registry(cwd):
    p = Path(cwd) / "diagnostics_outputs" / "model_skip_registry.json"
    return json.loads(p.read_text()) if p.exists() else {}


def flag_lines(cwd):
    p = Path(cwd) / thc.WHY_REPORT_FLAGS_PATH
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def check(mod, cwd, report):
    write_json(Path(cwd) / "diagnostics_outputs" / "training_diagnostics.json", report)
    return mod.check_training_health()


MPO_RUNNER = r'''
import json, os, subprocess, sys
code_root, worktree = sys.argv[1], sys.argv[2]
sys.path[:0] = [code_root, worktree]
import meta_prediction_optimizer_anti_overfit as mpo
calls = []
def fake_run(cmd, env=None, stdout=None, stderr=None, **kw):
    calls.append(cmd[cmd.index("--model-type") + 1])
    os.makedirs("models/reinforcement", exist_ok=True)
    with open("models/reinforcement/best_model.meta.json", "w") as f:
        json.dump({"training_metrics": {"r2": 0.1}}, f)
    return subprocess.CompletedProcess(cmd, 0)
subprocess.run = fake_run
err = None
try:
    mpo._s88_run_compare_models({"survivors": "s.json", "lottery_data": "l.json", "trials": 1,
                                 "enable_diagnostics": False})
except Exception as e:
    err = f"{type(e).__name__}: {e}"
print("MPO_RESULT=" + json.dumps({"calls": calls, "error": err}), flush=True)
'''


def run_mpo(code_root, cwd):
    """The compare optimizer's family list from code_root, run in cwd with subprocess.run faked."""
    runner = Path(cwd).parent / f"mpo_runner_{Path(cwd).name}.py"
    runner.write_text(MPO_RUNNER)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PYTHONPATH="")
    p = subprocess.run([sys.executable, "-B", str(runner), str(code_root), str(WORKTREE)], cwd=str(cwd),
                       capture_output=True, text=True, env=env, timeout=600)
    line = [l for l in p.stdout.splitlines() if l.startswith("MPO_RESULT=")]
    assert line, f"runner produced no result: rc={p.returncode}\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}"
    res = json.loads(line[-1][len("MPO_RESULT="):])
    res["stdout"] = p.stdout
    return res


# ─────────────────────────────────────────────────────────────────────────────
# (a) benching off
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("setting", ["absent", "off"])
def test_off_never_skipped_flag_raised(setting, tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    rules = {"max_consecutive_critical": 3, "skip_duration_hours": 24}
    if setting == "off":
        rules["benching"] = "off"
    policies(tmp_path, rules)
    caplog.set_level(logging.WARNING, logger=thc.logger.name)
    results = [check(thc, tmp_path, critical_report()) for _ in range(5)]
    actions = [r["action"] for r in results]
    assert actions == ["RETRY"] * 5, credit("OFF_THC")
    assert thc.is_model_skipped("neural_net") is False, credit("OFF_THC")
    assert registry(tmp_path)["neural_net"]["consecutive_critical"] == 5, credit("OFF_COUNT")
    st = thc.get_skip_status()["neural_net"]
    assert st["skipped"] is False and st["why_report_needed"] is True, credit("OFF_THC")
    # the flag: absent below the limit, present from the limit on, with family, count and timestamps
    assert [("why_report_needed" in r) for r in results] == [False, False, True, True, True], credit("FLAG_RESULT")
    f3 = results[2]["why_report_needed"]
    assert f3["model_type"] == "neural_net" and f3["consecutive_critical"] == 3 and f3["threshold"] == 3, \
        credit("FLAG_RESULT")
    assert f3["critical_since"] and f3["last_critical"] and f3["flagged_at"], credit("FLAG_RESULT")
    assert results[2]["consecutive_critical"] == 3, credit("FLAG_RESULT")
    lines = flag_lines(tmp_path)
    assert [l["consecutive_critical"] for l in lines] == [3, 4, 5], credit("FLAG_FILE")
    assert all(l["model_type"] == "neural_net" and l["schema"] == thc.WHY_REPORT_FLAG_SCHEMA for l in lines), \
        credit("FLAG_FILE")
    warns = [r for r in caplog.records if r.levelno == logging.WARNING and "[WHY-REPORT]" in r.getMessage()]
    assert len(warns) == 3 and "neural_net" in warns[0].getMessage(), credit("FLAG_LOG")


def test_off_multi_model_winner_flagged(tmp_path, monkeypatch):
    """The multi-model ('models') path carries the same flag for the winner."""
    monkeypatch.chdir(tmp_path)
    policies(tmp_path, {"max_consecutive_critical": 2})
    res = [check(thc, tmp_path, multi_report("catboost")) for _ in range(3)]
    assert [r["action"] for r in res] == ["RETRY"] * 3, credit("OFF_THC")
    assert res[1]["why_report_needed"]["model_type"] == "catboost", credit("FLAG_RESULT")
    assert len(flag_lines(tmp_path)) == 2, credit("FLAG_FILE")


def test_off_compare_reader_keeps_family(tmp_path):
    """meta_prediction_optimizer_anti_overfit: a family over the limit stays in the compare at benching off."""
    for setting in ("absent", "off", "on"):
        cwd = tmp_path / setting
        cwd.mkdir()
        rules = {"max_consecutive_critical": 3}
        if setting != "absent":
            rules["benching"] = setting
        policies(cwd, rules)
        write_json(cwd / "diagnostics_outputs" / "model_skip_registry.json",
                   {"neural_net": {"consecutive_critical": 7, "last_critical": "2026-11-26T00:00:00+00:00"}})
        r = run_mpo(CODE_ROOT, cwd)
        assert r["error"] is None, r
        if setting == "on":   # non-vacuity: the same registry benches it with the switch on
            assert "neural_net" not in r["calls"] and len(r["calls"]) == 3, credit("ON_MPO")
        else:
            assert sorted(r["calls"]) == sorted(FAMILIES), credit("OFF_MPO")
            assert "[S90][WHY-REPORT]" in r["stdout"] and "neural_net" in r["stdout"], credit("OFF_MPO_WHY")


# ─────────────────────────────────────────────────────────────────────────────
# (b) benching on + until_reset == BASE_REV
# ─────────────────────────────────────────────────────────────────────────────

def _norm_registry(reg):
    return {m: {k: (bool(v) if k == "last_critical" else v) for k, v in e.items()} for m, e in reg.items()}


def _norm_status(st):
    return {m: {k: (bool(v) if k == "last_critical" else v) for k, v in e.items()} for m, e in st.items()}


def _history(cwd):
    h = Path(cwd) / "diagnostics_outputs" / "history"
    out = []
    for p in sorted(h.glob("*.json")):
        d = json.loads(p.read_text())
        d.pop("archived_at", None)
        out.append(d)
    return out


def _sequence(mod, cwd, rules, monkeypatch):
    """critical x4 (single), reset, critical, multi-model critical x3; every observable after each call."""
    Path(cwd).mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(cwd)
    if rules is not None:
        policies(cwd, rules)
    obs = []

    def snap(r):
        obs.append({"result": r, "registry": _norm_registry(registry(cwd)),
                    "skipped": {m: mod.is_model_skipped(m) for m in FAMILIES},
                    "status": _norm_status(mod.get_skip_status())})

    for _ in range(4):
        snap(check(mod, cwd, critical_report()))
    mod.reset_skip_registry("neural_net")
    snap(None)
    snap(check(mod, cwd, critical_report()))
    for _ in range(3):
        snap(check(mod, cwd, multi_report("catboost")))
    obs.append({"history": _history(cwd), "flags": flag_lines(cwd)})
    return obs


@pytest.mark.parametrize("new_rules", [{"benching": "on"},
                                       {"benching": "on", "expiry_mode": "until_reset",
                                        "max_consecutive_critical": 3, "skip_duration_hours": 24}])
def test_on_until_reset_equals_base(new_rules, base_dir, tmp_path, monkeypatch):
    base = load_module("thc_base_main", base_dir / "training_health_check.py")
    a = _sequence(base, tmp_path / "base", None, monkeypatch)              # BASE_REV, today's policy (no key)
    b = _sequence(thc, tmp_path / "new", new_rules, monkeypatch)
    assert any(o.get("result") and o["result"]["action"] == "SKIP_MODEL" for o in a), "vacuous: base never skips"
    assert a == b, credit("ON_IDENTITY")


def test_on_compare_reader_equals_base(base_dir, tmp_path):
    reg = {"neural_net": {"consecutive_critical": 3, "last_critical": "2026-11-20T00:00:00+00:00"},
           "xgboost": {"consecutive_critical": 2, "last_critical": "2026-11-20T00:00:00+00:00"},
           "catboost": {"consecutive_critical": 9, "last_critical": None}}
    got = {}
    for name, root, rules in (("base", base_dir, None), ("new", CODE_ROOT, {"benching": "on"})):
        cwd = tmp_path / name
        cwd.mkdir()
        if rules is not None:
            policies(cwd, rules)
        write_json(cwd / "diagnostics_outputs" / "model_skip_registry.json", reg)
        got[name] = run_mpo(root, cwd)
    assert sorted(got["base"]["calls"]) == ["lightgbm", "xgboost"], "vacuous: base benches nothing"
    assert got["base"]["calls"] == got["new"]["calls"] and got["base"]["error"] == got["new"]["error"], \
        credit("ON_MPO_IDENTITY")


# ─────────────────────────────────────────────────────────────────────────────
# (c) hours: the count restarts after a timed skip expires
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("age_hours,expect", [(25, ["RETRY", "RETRY", "SKIP_MODEL"]),   # expired: restart
                                              (1, ["SKIP_MODEL", "SKIP_MODEL", "SKIP_MODEL"])])  # still active
def test_hours_restart(age_hours, expect, base_dir, tmp_path, monkeypatch):
    rules = {"benching": "on", "expiry_mode": "hours", "skip_duration_hours": 24, "max_consecutive_critical": 3}
    last = (datetime.now(timezone.utc) - timedelta(hours=age_hours)).isoformat()
    runs = {}
    for name, mod in (("new", thc), ("base", load_module("thc_base_h", base_dir / "training_health_check.py"))):
        cwd = tmp_path / name
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        policies(cwd, rules)
        write_json(cwd / "diagnostics_outputs" / "model_skip_registry.json",
                   {"neural_net": {"consecutive_critical": 3, "last_critical": last}})
        assert mod.is_model_skipped("neural_net") is (age_hours < 24)
        runs[name] = [(r["action"], registry(cwd)["neural_net"]["consecutive_critical"])
                      for r in (check(mod, cwd, critical_report()) for _ in range(3))]
    assert [a for a, _ in runs["new"]] == expect, credit("HOURS_RESTART")
    if age_hours > 24:
        assert [c for _, c in runs["new"]] == [1, 2, 3], credit("HOURS_RESTART")
        # non-vacuity: BASE_REV re-skips at once (the recorded defect, READY_LOOP_DIAG_2_20261123.md:229)
        assert [a for a, _ in runs["base"]] == ["SKIP_MODEL"] * 3
    else:
        assert runs["new"] == runs["base"], credit("HOURS_ACTIVE_IDENTITY")


# ─────────────────────────────────────────────────────────────────────────────
# (d) a bad value fails loudly
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad", ["OFF", "true", True, 0, None, "bench"])
def test_bad_value_raises(bad, tmp_path, monkeypatch):
    rules = {"benching": bad}
    for call in (lambda: thc.benching_mode(rules),
                 lambda: thc.skip_active({"consecutive_critical": 0}, rules),
                 lambda: thc._wave2_settings({"model_skip_rules": rules})):
        try:
            call()
        except ValueError as e:
            assert "benching" in str(e), credit("BAD_VALUE")
        else:
            pytest.fail(credit("BAD_VALUE") + f" no ValueError for benching={bad!r}")
    monkeypatch.chdir(tmp_path)
    policies(tmp_path, rules)
    r = check(thc, tmp_path, critical_report())   # like expiry_mode: module error, nothing counted
    assert r["action"] == "PROCEED_WITH_NOTE" and r["severity"] == "absent", credit("BAD_VALUE")
    assert any("benching" in i for i in r["issues"]), credit("BAD_VALUE")
    assert registry(tmp_path) == {} and flag_lines(tmp_path) == [], credit("BAD_VALUE")


def test_bad_value_compare_reader_raises(tmp_path):
    policies(tmp_path, {"benching": "maybe"})
    write_json(tmp_path / "diagnostics_outputs" / "model_skip_registry.json",
               {"neural_net": {"consecutive_critical": 1, "last_critical": None}})
    r = run_mpo(CODE_ROOT, tmp_path)
    assert r["error"] and "benching" in r["error"] and r["calls"] == [], credit("BAD_VALUE_MPO")


# ─────────────────────────────────────────────────────────────────────────────
# (e) isolation (runs last)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# (f) package review fixes, 2026-11-26: flag_id, critical_since_known, fallback reader, fixed path, G-2
# ─────────────────────────────────────────────────────────────────────────────

def test_flag_id_stable_and_since_known(tmp_path, monkeypatch):
    """(b, c) a run of criticals that started on record: every repeat flag carries one flag_id, built from
    the family and critical_since, and critical_since_known is True. A new run of criticals (after a reset)
    gets a new flag_id."""
    monkeypatch.chdir(tmp_path)
    policies(tmp_path, {"max_consecutive_critical": 3})
    res = [check(thc, tmp_path, critical_report()) for _ in range(5)]
    lines = flag_lines(tmp_path)
    since = registry(tmp_path)["neural_net"]["critical_since"]
    assert {l["flag_id"] for l in lines} == {f"neural_net@{since}"}, credit("FLAG_ID")
    assert res[4]["why_report_needed"]["flag_id"] == f"neural_net@{since}", credit("FLAG_ID")
    assert all(l["critical_since_known"] is True and l["critical_since"] == since for l in lines), \
        credit("SINCE_KNOWN")
    thc.reset_skip_registry("neural_net")
    assert "critical_since" not in registry(tmp_path)["neural_net"], credit("FLAG_ID_NEW_RUN")
    for _ in range(3):
        check(thc, tmp_path, critical_report())
    ids = [l["flag_id"] for l in flag_lines(tmp_path)]
    assert len(set(ids)) == 2 and ids[-1] != ids[0], credit("FLAG_ID_NEW_RUN")


def test_since_unknown_for_a_count_that_predates_the_record(tmp_path, monkeypatch):
    """(b) an entry that already has a count (written before critical_since existed, or under benching on):
    critical_since stays None, critical_since_known is False, and (c) the flag_id is anchored on the first
    flagged_at of this run of criticals, so the repeats still share it."""
    monkeypatch.chdir(tmp_path)
    policies(tmp_path, {"max_consecutive_critical": 3})
    write_json(tmp_path / "diagnostics_outputs" / "model_skip_registry.json",
               {"neural_net": {"consecutive_critical": 2, "last_critical": "2026-11-20T00:00:00+00:00"}})
    res = [check(thc, tmp_path, critical_report()) for _ in range(3)]
    lines = flag_lines(tmp_path)
    assert [l["consecutive_critical"] for l in lines] == [3, 4, 5]
    assert all(l["critical_since"] is None and l["critical_since_known"] is False for l in lines), \
        credit("SINCE_KNOWN")
    assert res[0]["why_report_needed"]["critical_since_known"] is False, credit("SINCE_KNOWN")
    first = registry(tmp_path)["neural_net"].get("why_flag_first_at")
    assert first, credit("FLAG_ID")
    assert {l["flag_id"] for l in lines} == {f"neural_net@{first}"}, credit("FLAG_ID")
    assert first == lines[0]["last_critical"], credit("FLAG_ID")


def _stub_thc_root(tmp_path):
    """A code root with this compare reader and a training_health_check that cannot be imported."""
    root = tmp_path / "code_no_thc"
    root.mkdir()
    (root / "meta_prediction_optimizer_anti_overfit.py").write_bytes(
        (CODE_ROOT / "meta_prediction_optimizer_anti_overfit.py").read_bytes())
    (root / "training_health_check.py").write_text("raise ImportError('synthetic: health check unavailable')\n")
    return root


@pytest.mark.parametrize("setting", ["absent", "off", "on", "bad"])
def test_compare_reader_fallback_without_health_check(setting, tmp_path):
    """(d) the compare reader's own fallback, used when training_health_check cannot be imported."""
    root = _stub_thc_root(tmp_path)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    rules = {"max_consecutive_critical": 3}
    if setting != "absent":
        rules["benching"] = "OFF" if setting == "bad" else setting
    policies(cwd, rules)
    write_json(cwd / "diagnostics_outputs" / "model_skip_registry.json",
               {"neural_net": {"consecutive_critical": 4, "last_critical": "2026-11-26T00:00:00+00:00"},
                "catboost": {"consecutive_critical": 1, "last_critical": "2026-11-26T00:00:00+00:00"}})
    r = run_mpo(root, cwd)
    if setting == "bad":
        assert r["error"] and "ValueError" in r["error"] and "benching" in r["error"], credit("FALLBACK")
        return
    assert r["error"] is None, r
    if setting == "on":
        assert "neural_net" not in r["calls"] and len(r["calls"]) == 3, credit("FALLBACK")
        assert "[S90][WHY-REPORT]" not in r["stdout"], credit("FALLBACK")
    else:
        assert sorted(r["calls"]) == sorted(FAMILIES), credit("FALLBACK")
        assert "[S90][WHY-REPORT]" in r["stdout"] and "'neural_net'" in r["stdout"], credit("FALLBACK")
        assert "'catboost'" not in r["stdout"].split("[S90][WHY-REPORT]")[1].splitlines()[0]


def _fixed_report(mt, critical, tag):
    """Graded by grading_rules fixed from the training summary: 1/17 and val/train 2.0 grade critical."""
    best, vl = (1, 2.0) if critical else (14, 1.05)
    return {"model_type": mt, "status": "complete",
            "training_summary": {"best_val_round": best, "rounds_captured": 17, "final_train_loss": 1.0,
                                 "final_val_loss": vl},
            "diagnosis": {"severity": "critical" if critical else "ok", "issues": [f"{tag}-ISSUE"],
                          "suggested_fixes": [f"{tag}-FIX"]}}


def _fixed_fixture(cwd, winner_critical, nn_critical, **td):
    d = Path(cwd) / "diagnostics_outputs"
    write_json(d / "catboost_diagnostics.json", _fixed_report("catboost", winner_critical, "CATBOOST"))
    write_json(d / "neural_net_diagnostics.json", _fixed_report("neural_net", nn_critical, "NN"))
    write_json(d / "training_diagnostics.json", _fixed_report("neural_net", nn_critical, "NN"))
    write_json(d / "compare_models_summary_latest.json",
               {"run_id": "R1", "models": {"catboost": {}, "neural_net": {}},
                "winner_best_effort": {"model_type": "catboost", "score": 0.1}})
    write_json(Path(cwd) / "watcher_policies.json",
               {"training_diagnostics": dict(td, grading_rules="fixed",
                                             model_skip_rules={"max_consecutive_critical": 2})})


def test_fixed_grading_at_limit_benching_off(tmp_path, monkeypatch):
    """(d) grading_rules fixed, the winner critical, benching off: RETRY at and over the limit, the flag on
    the result and in the file, the family named is the winner."""
    monkeypatch.chdir(tmp_path)
    _fixed_fixture(tmp_path, winner_critical=True, nn_critical=False)
    res = [thc.check_training_health() for _ in range(3)]
    assert [r["grading_rules"] for r in res] == ["fixed"] * 3
    assert [r["action"] for r in res] == ["RETRY"] * 3, credit("FIXED_FLAG")
    assert [("why_report_needed" in r) for r in res] == [False, True, True], credit("FIXED_FLAG")
    assert res[1]["why_report_needed"]["model_type"] == "catboost", credit("FIXED_FLAG")
    assert res[1]["consecutive_critical"] == 2, credit("FIXED_FLAG")
    assert [l["model_type"] for l in flag_lines(tmp_path)] == ["catboost", "catboost"], credit("FIXED_FLAG")
    assert thc.is_model_skipped("catboost") is False


def test_g2_promotion_plus_why_flag(tmp_path, monkeypatch):
    """(d) G-2 and benching together: grade_scope all_families promotes the critical neural_net over an ok
    winner; the result reports neural_net's own report (G-2) and the flag is raised for neural_net, the
    family that set the grade, not for the winner."""
    monkeypatch.chdir(tmp_path)
    _fixed_fixture(tmp_path, winner_critical=False, nn_critical=True, grade_scope="all_families")
    res = [thc.check_training_health() for _ in range(2)]
    r = res[1]
    assert r["model_type"] == "neural_net" and r["grade_set_by"] == "neural_net", credit("G2_FLAG")
    assert r["winner_model_type"] == "catboost" and r["suggested_fixes"] == ["NN-FIX"], credit("G2_FLAG")
    assert r["issues"][0] == "NN-ISSUE" and "[catboost] CATBOOST-ISSUE" in r["issues"], credit("G2_FLAG")
    assert r["action"] == "RETRY" and r["why_report_needed"]["model_type"] == "neural_net", credit("G2_FLAG")
    assert r["why_report_needed"]["flag_id"].startswith("neural_net@"), credit("G2_FLAG")
    reg = registry(tmp_path)
    assert reg["neural_net"]["consecutive_critical"] == 2 and "catboost" not in reg, credit("G2_FLAG")
    assert [l["model_type"] for l in flag_lines(tmp_path)] == ["neural_net"], credit("G2_FLAG")


def test_interrupted_registry_write_keeps_flag_identity(tmp_path, monkeypatch):
    """R1 F1 (Beta Q5): a registry write interrupted part-way leaves the previous registry intact, so one
    run of criticals keeps one flag_id. The interruption is injected into json.dump (part of the JSON is
    written, then OSError), for the check and for reset_skip_registry."""
    import types
    monkeypatch.chdir(tmp_path)
    policies(tmp_path, {"max_consecutive_critical": 3})
    for _ in range(3):
        check(thc, tmp_path, critical_report())
    fid = flag_lines(tmp_path)[0]["flag_id"]
    reg_p = tmp_path / "diagnostics_outputs" / "model_skip_registry.json"
    before = reg_p.read_bytes()

    def interrupted(obj, f, **kw):
        f.write(json.dumps(obj, **kw)[:25])
        raise OSError("synthetic interruption mid-write")
    fake_json = types.SimpleNamespace(**{k: getattr(json, k) for k in dir(json) if not k.startswith("__")})
    fake_json.dump = interrupted
    monkeypatch.setattr(thc, "json", fake_json)
    check(thc, tmp_path, critical_report())                       # its registry write is interrupted
    assert reg_p.read_bytes() == before, credit("ATOMIC")
    assert [p.name for p in reg_p.parent.iterdir() if p.name.endswith(".tmp")] == [], credit("ATOMIC")
    thc.reset_skip_registry("neural_net")                         # an interrupted reset
    assert reg_p.read_bytes() == before, credit("ATOMIC_RESET")
    monkeypatch.setattr(thc, "json", json)
    for _ in range(3):
        check(thc, tmp_path, critical_report())
    assert {l["flag_id"] for l in flag_lines(tmp_path)} == {fid}, credit("ATOMIC")
    assert registry(tmp_path)["neural_net"]["consecutive_critical"] == 6
    # non-vacuity: a real reset still writes (and ends the run of criticals)
    thc.reset_skip_registry("neural_net")
    assert registry(tmp_path)["neural_net"]["consecutive_critical"] == 0


@pytest.mark.parametrize("age", ["expired", "active", "unreadable"])
def test_compare_fallback_honours_hours_expiry(age, tmp_path):
    """R1 F5 (Beta Q3): with benching on and expiry_mode hours, the compare reader's import-failure
    fallback lets an expired timed skip go (the family is trained again), keeps an active one, and keeps
    a skip whose last_critical cannot be read (as skip_active)."""
    root = _stub_thc_root(tmp_path)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    policies(cwd, {"max_consecutive_critical": 3, "benching": "on", "expiry_mode": "hours",
                   "skip_duration_hours": 24})
    last = {"expired": (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat(),
            "active": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
            "unreadable": "not a time"}[age]
    write_json(cwd / "diagnostics_outputs" / "model_skip_registry.json",
               {"neural_net": {"consecutive_critical": 3, "last_critical": last}})
    r = run_mpo(root, cwd)
    assert r["error"] is None, r
    if age == "expired":
        assert sorted(r["calls"]) == sorted(FAMILIES), credit("FALLBACK_HOURS")
    else:
        assert "neural_net" not in r["calls"] and len(r["calls"]) == 3, credit("FALLBACK_HOURS")


def test_snapshot_detects_a_write(tmp_path):
    """Non-vacuity of the isolation gate: a write in a watched place, or at the top level, is seen.
    (No git repository is created for this; the git-status part is the worktree's own, read only.)"""
    repo = tmp_path / "fake_repo"
    (repo / "data").mkdir(parents=True)
    (repo / "catboost_info").mkdir()
    before = _snapshot(repo, None)
    (repo / "data" / "x.json").write_text("{}")
    (repo / "catboost_info" / "learn_error.tsv").write_text("x")
    (repo / "stray.txt").write_text("x")
    after = _snapshot(repo, None)
    assert before["watched"]["data"] != after["watched"]["data"]
    assert before["watched"]["catboost_info"] != after["watched"]["catboost_info"]
    assert before["top"] != after["top"]


# The one declared exception: the file this pytest run's own stdout is redirected to (BENCH_RUN_LOG, one file,
# relative to the worktree), because a log under data/ is written while the suite runs.
RUN_LOG = os.environ.get("BENCH_RUN_LOG")


def _diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += _diff(a.get(k), b.get(k), f"{path}/{k}")
        return out
    return [] if a == b else [path]


def test_zz_nothing_written_outside_basetemp():
    after = _snapshot(WORKTREE, _BEFORE["basetemp"])
    before = _BEFORE["snap"]
    changed = _diff({k: before[k] for k in ("top", "watched")}, {k: after[k] for k in ("top", "watched")})
    changed += [f"git:{l}" for l in sorted(set(before["git"]) ^ set(after["git"]))]
    if RUN_LOG:
        changed = [c for c in changed if c not in (f"/watched/data/{Path(RUN_LOG).relative_to('data')}",
                                                   f"git:!! {RUN_LOG}")]
    assert changed == [], credit("ISOLATION") + f" changed: {changed[:20]}"
