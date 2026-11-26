#!/usr/bin/env python3
"""
G-2 fix tests (2026-11-26): training_health_check._check_training_health_fixed with
grading_rules fixed + grade_scope all_families, when a worse non-winner family is promoted to
set the grade. Found by Gemini trial test 2, 2026-11-26. CPU only; every file is written under
pytest's tmp_path (each check chdirs there, the module uses relative paths).

  * promoted case: issues, suggested_fixes, metrics, archive file name and content are the
    promoted family's own; the winner's issues appear prefixed; the result names both families.
  * unchanged modes: grading_rules legacy (default) and fixed + grade_scope winner give results
    and archive entries identical to the function at BASE_REV (git show), with a non-vacuity
    check that the comparison can see a difference (the promoted case differs from BASE_REV).
  * mutants: each restores one piece of the old code in a temp copy and runs one named test
    in a subprocess against it; that test must go red.

  * R1 F4 (Beta X-BENCHING-SWITCH R1 Q4): in the promoted case report_path, report_source and freshness
    are the grade-setting family's own, the winner's under winner_*; the optional health log the same.

Run (from the worktree root):
    PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= python3 -B -m pytest -q tests/test_g2_grade_scope_fix.py -p no:cacheprovider
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import contextlib
import glob
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

WORKTREE = Path(__file__).resolve().parents[1]
BASE_REV = "7f28109"   # main the package is rebased on (933a4a0 -> 7f28109 changed none of its files)
CODE_PATH = Path(os.environ.get("G2_THC_PATH", str(WORKTREE / "training_health_check.py")))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


thc = _load(CODE_PATH, "thc_under_test")


@pytest.fixture(scope="module")
def thc_base(tmp_path_factory):
    src = subprocess.run(["git", "-C", str(WORKTREE), "show", f"{BASE_REV}:training_health_check.py"],
                         check=True, capture_output=True, text=True).stdout
    p = tmp_path_factory.mktemp("base") / "thc_base.py"
    p.write_text(src)
    return _load(p, "thc_base")


@contextlib.contextmanager
def chdir(d):
    old = os.getcwd()
    os.chdir(d)
    try:
        yield
    finally:
        os.chdir(old)


def _report(mt, sev, best, rounds, tl, vl, tag):
    return {"model_type": mt, "status": "complete",
            "training_summary": {"best_val_round": best, "rounds_captured": rounds,
                                 "final_train_loss": tl, "final_val_loss": vl},
            "diagnosis": {"severity": sev, "issues": [f"{tag}-ISSUE"], "suggested_fixes": [f"{tag}-FIX"]}}


def _wj(p, o):
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "w") as f:
        json.dump(o, f)


def setup_fixture(root, policies, nn_bad=True):
    """Winner catboost grades ok (14/17, overfit 1.05). neural_net grades critical (1/17, 2.0)
    when nn_bad, else ok. The canonical file holds the NN report (NN wrote last)."""
    d = os.path.join(root, "diagnostics_outputs")
    win = _report("catboost", "ok", 14, 17, 1.0, 1.05, "CATBOOST")
    nn = (_report("neural_net", "critical", 1, 17, 1.0, 2.0, "NN") if nn_bad
          else _report("neural_net", "ok", 15, 17, 1.0, 1.02, "NN"))
    _wj(os.path.join(d, "catboost_diagnostics.json"), win)
    _wj(os.path.join(d, "neural_net_diagnostics.json"), nn)
    _wj(os.path.join(d, "training_diagnostics.json"), nn)
    _wj(os.path.join(d, "compare_models_summary_latest.json"),
        {"run_id": "R1", "models": {"catboost": {}, "neural_net": {}},
         "winner_best_effort": {"model_type": "catboost", "score": 0.1}})
    _wj(os.path.join(root, "watcher_policies.json"), {"training_diagnostics": policies})


def run_check(mod, root, policies, nn_bad=True):
    os.makedirs(root, exist_ok=True)
    with chdir(root):
        setup_fixture(root, policies, nn_bad)
        r = mod.check_training_health()
        arch = []
        for a in sorted(glob.glob(os.path.join(root, "diagnostics_outputs/history/*.json"))):
            with open(a) as f:
                o = json.load(f)
            arch.append((os.path.basename(a).rsplit("_", 2)[0], o))
    return r, arch


ALL_FAMILIES = {"grading_rules": "fixed", "grade_scope": "all_families"}


@pytest.fixture(scope="module")
def promoted(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("promoted"))
    r, arch = run_check(thc, root, ALL_FAMILIES)
    # non-vacuity: the non-winner really sets the grade, the winner really grades ok
    assert r["model_type"] == "neural_net" and r["severity"] == "critical", r
    assert r["per_family"]["catboost"]["severity"] == "ok"
    return r, arch


# ── promoted family case ────────────────────────────────────────────────────

def test_promoted_issues_are_own_and_winner_prefixed(promoted):
    r, _ = promoted
    iss = r["issues"]
    assert iss[0] == "NN-ISSUE", iss                                  # promoted family's own, first
    assert any(i.startswith("WATCHER: Early stop ratio 0.06") for i in iss), iss
    assert "CATBOOST-ISSUE" not in iss, iss                           # winner's never unprefixed
    assert iss.count("[catboost] CATBOOST-ISSUE") == 1, iss           # winner's once, prefixed
    assert not any(i.startswith("[neural_net]") for i in iss), iss    # own issues not duplicated
    assert all(i.startswith("[catboost] ") for i in iss if "freshness" in i), iss


def test_promoted_fixes_are_own(promoted):
    r, _ = promoted
    assert r["suggested_fixes"] == ["NN-FIX"], r["suggested_fixes"]


def test_promoted_metrics_are_own(promoted):
    r, _ = promoted
    assert abs(r["metrics"]["early_stop_ratio"] - 1 / 17) < 1e-9, r["metrics"]
    assert r["metrics"]["overfit_ratio"] == 2.0, r["metrics"]


def test_promoted_archive_is_own(promoted):
    r, arch = promoted
    assert len(arch) == 1, arch
    name, o = arch[0]
    assert name == "neural_net"
    assert o["diagnostics"]["model_type"] == "neural_net"
    assert o["diagnostics"]["diagnosis"]["suggested_fixes"] == ["NN-FIX"]
    assert o["watcher_severity"] == "critical"
    assert o["watcher_issues"] == r["issues"]


def test_promoted_result_names_winner_and_grade_setter(promoted):
    r, _ = promoted
    assert r["winner_model_type"] == "catboost"
    assert r["grade_set_by"] == "neural_net"
    assert "neural_net" in r["grade_note"] and "catboost" in r["grade_note"]


def test_promoted_case_differs_from_base(promoted, thc_base, tmp_path):
    """Non-vacuity for the equivalence tests: on the promoted case the comparison sees a change."""
    rb, ab = run_check(thc_base, str(tmp_path / "b"), ALL_FAMILIES)
    r, a = promoted
    assert _norm(rb) != _norm(r)
    assert ab[0][1]["diagnostics"]["model_type"] == "catboost"      # base archived the winner


def test_promoted_provenance_is_grade_setters(tmp_path):
    """R1 F4, Beta Q4's fixture: stale_report_action grade, the winner's report stale (another run id),
    the promoted family's fresh. Consumers get the grade setter's report path, source and freshness
    together; the winner's are kept under winner_*; the optional health log carries the same."""
    root = str(tmp_path)
    pol = dict(ALL_FAMILIES, stale_report_action="grade", training_health_log="on")
    os.makedirs(root, exist_ok=True)
    with chdir(root):
        setup_fixture(root, pol)
        d = os.path.join(root, "diagnostics_outputs")
        for fn, rid in (("catboost_diagnostics.json", "OLD"), ("neural_net_diagnostics.json", "R1")):
            with open(os.path.join(d, fn)) as f:
                o = json.load(f)
            _wj(os.path.join(d, fn), dict(o, diagnostics_run_id=rid))
        with open(os.path.join(d, "compare_models_summary_latest.json")) as f:
            o = json.load(f)
        _wj(os.path.join(d, "compare_models_summary_latest.json"), dict(o, diagnostics_run_id="R1"))
        r = thc.check_training_health()
        with open(os.path.join(d, "training_health_log.jsonl")) as f:
            log = [json.loads(l) for l in f if l.strip()]
    # non-vacuity: the winner really is stale and graded, the NN really is promoted
    assert r["grade_set_by"] == "neural_net" and r["winner_model_type"] == "catboost", r
    assert r["report_path"].endswith("neural_net_diagnostics.json"), r["report_path"]
    assert r["report_source"] == "all_families_grade_setter_report", r["report_source"]
    assert r["freshness"]["freshness"] == "fresh" and r["freshness"]["report_run_id"] == "R1", r["freshness"]
    assert r["winner_report_path"].endswith("catboost_diagnostics.json"), r
    assert r["winner_report_source"] == "compare_pointer_winner_report", r
    assert r["winner_freshness"]["freshness"] == "stale" and r["winner_freshness"]["report_run_id"] == "OLD"
    assert len(log) == 1, log
    L = log[0]
    assert L["report_path"].endswith("neural_net_diagnostics.json") and L["freshness"] == "fresh", L
    assert L["report_run_id"] == "R1" and L["grade_set_by"] == "neural_net", L
    assert L["winner_report_path"].endswith("catboost_diagnostics.json") and L["winner_freshness"] == "stale", L


# ── unchanged modes: identical to BASE_REV ──────────────────────────────────

def _norm(r):
    return json.loads(json.dumps(r, sort_keys=True, default=str))


def _norm_arch(arch):
    return [(n, {k: v for k, v in o.items() if k != "archived_at"}) for n, o in arch]


def _registry(root):
    p = os.path.join(root, "diagnostics_outputs/model_skip_registry.json")
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        reg = json.load(f)
    # [PACKAGE 2026-11-26] critical_since / why_flag_first_at are the benching switch's registry keys
    # (default off), tested in tests/test_benching_switch.py; this suite compares the G-2 change only
    return {m: {k: v for k, v in e.items() if k not in ("last_critical", "critical_since", "why_flag_first_at")}
            for m, e in reg.items()}


@pytest.mark.parametrize("policies", [{}, {"grading_rules": "legacy"}, {"grading_rules": "fixed"},
                                      {"grading_rules": "fixed", "grade_scope": "winner"}],
                         ids=["legacy_default", "legacy_explicit", "fixed_default_scope", "fixed_winner"])
@pytest.mark.parametrize("nn_bad", [True, False], ids=["nn_critical", "nn_ok"])
def test_unchanged_modes_identical_to_base(policies, nn_bad, thc_base, tmp_path):
    rn, an = run_check(thc, str(tmp_path / "new"), policies, nn_bad)
    rb, ab = run_check(thc_base, str(tmp_path / "base"), policies, nn_bad)
    assert _norm(rn) == _norm(rb)
    assert _norm_arch(an) == _norm_arch(ab) and len(an) == 1
    assert _registry(str(tmp_path / "new")) == _registry(str(tmp_path / "base"))
    # non-vacuity: the fixture exercises what the fix touches in these modes
    if policies.get("grading_rules") == "fixed":
        assert set(rn["per_family"]) == {"catboost", "neural_net"} and rn["model_type"] == "catboost"
        assert "[neural_net] NN-ISSUE" in rn["issues"] and "CATBOOST-ISSUE" in rn["issues"]
        assert (rn["per_family"]["neural_net"]["severity"] == "critical") == nn_bad
        assert "grade_set_by" not in rn and "winner_model_type" not in rn
    else:
        assert "per_family" not in rn and rn["model_type"] == "neural_net"
        assert rn["suggested_fixes"] == ["NN-FIX"]


def test_all_families_without_promotion_identical_to_base(thc_base, tmp_path):
    """all_families with nothing worse than the winner: no promotion, same as BASE_REV."""
    rn, an = run_check(thc, str(tmp_path / "new"), ALL_FAMILIES, nn_bad=False)
    rb, ab = run_check(thc_base, str(tmp_path / "base"), ALL_FAMILIES, nn_bad=False)
    assert rn["model_type"] == "catboost" and "grade_set_by" not in rn
    assert _norm(rn) == _norm(rb) and _norm_arch(an) == _norm_arch(ab)


# ── mutants ─────────────────────────────────────────────────────────────────

MUTANTS = {
    "old_skip_condition": (
        [("        if m != model_type and (promoted is None or m != winner_name):",
          "        if m != model_type:")],
        "test_promoted_issues_are_own_and_winner_prefixed"),
    "winner_fixes": (
        [("'suggested_fixes': list(out_diag.get('diagnosis', {}).get('suggested_fixes', [])),",
          "'suggested_fixes': list(diag.get('diagnosis', {}).get('suggested_fixes', [])),")],
        "test_promoted_fixes_are_own"),
    "winner_archive": (
        [("    _archive_diagnostics(out_diag, severity, issues, model_type)",
          "    _archive_diagnostics(diag, severity, issues, model_type)")],
        "test_promoted_archive_is_own"),
    "winner_metrics": (
        [("'metrics': out_g.get('metrics'),", "'metrics': g.get('metrics'),")],
        "test_promoted_metrics_are_own"),
    # R1 F4
    "winner_report_path": (
        [("                       'report_path': per_family[promoted]['path'],",
          "                       'report_path': base['report_path'],")],
        "test_promoted_provenance_is_grade_setters"),
    "winner_freshness": (
        [("                       'freshness': family_reports[promoted][2] if promoted in family_reports else None})",
          "                       'freshness': base.get('freshness')})")],
        "test_promoted_provenance_is_grade_setters"),
    "health_log_no_winner_fields": (
        [("                                            'winner_report_source')} if result.get('grade_set_by') else {}),",
          "                                            'winner_report_source')} if False else {}),")],
        "test_promoted_provenance_is_grade_setters"),
}


@pytest.mark.skipif(os.environ.get("G2_THC_PATH") is not None, reason="inside a mutant run")
@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_mutant_turns_named_test_red(name, tmp_path):
    reps, target = MUTANTS[name]
    src = (WORKTREE / "training_health_check.py").read_text()
    for a, b in reps:
        assert src.count(a) == 1, f"mutant {name}: anchor not found once: {a!r}"
        src = src.replace(a, b)
    mp = tmp_path / "thc_mutant.py"
    mp.write_text(src)
    env = dict(os.environ, G2_THC_PATH=str(mp), CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp",
           str(tmp_path / "pt"), f"{__file__}::{target}"]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=str(tmp_path))
    assert p.returncode == 1 and "1 failed" in p.stdout, f"mutant {name} survived:\n{p.stdout[-2000:]}"
