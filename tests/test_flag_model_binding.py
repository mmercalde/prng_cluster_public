#!/usr/bin/env python3
"""
tests/test_flag_model_binding.py - unit W3 (2026-11-27): the why-report flag names the model that was flagged,
with proof. Design: docs/drafts/unit_w/DESIGN_FLAG_MODEL_BINDING_20261127_REV2.md. CPU only, synthetic files,
stubbed children. Run with pytest and a --basetemp under <this tree>/data/.

Gates (one test each; mutants below must turn the named gate red):
  W3.bound_nn_legacy           NN pair published, critical canonical report of this run: one flag line, bound to the
                               staged checkpoint (path, dir, sha), report sha = graded bytes; models_dir from WATCHER
                               ignored (relocated dir)
  W3.strict_stale              WATCHER id given: a report with another id, or with no id and a post-start mtime, is
                               stale, not graded, no flag line; the skip count does not move (owner decision 2)
  W3.report_sha_link           same run id, same family, a substituted report (e.g. a trial report stamped with the id):
                               unbound
  W3.staged_ck_changed         staged checkpoint changed after publication: unbound, never another sha
  W3.nn_report_ck              NN report's checkpoint_sha256 differs from the record: unbound
  W3.override_family           leftover compare_models_summary.json renames the family (S93): unbound
  W3.unbound_tree_legacy       tree family with no report (level legacy): unbound with the reason
  W3.fixed_promoted            fixed + all_families, healthy winner, critical non-winner: flag bound to the promoted
                               family's own report and checkpoint (Beta Q2c, G-2)
  W3.fixed_leftover_pointer    fixed grading, a leftover pointer of another run next to this run's report: the current
                               report is graded, not 'stale' (Beta Q5)
  W3.report_swapped            report rewritten between grading and binding: the graded bytes decide
  W3.read_write_errors         identity record unreadable, checkpoint unreadable: one line, unbound, same action
  W3.deadline                  binding I/O stalls: flag appended once within the deadline, unbound, reason 'deadline';
                               invalid deadline settings use 10 s and say so
  W3.publish_recheck           staged checkpoint changed between pending and publish: not published
  W3.nn_writer_pair            real train_single_trial.py CLI (CPU, tiny NPZ) with a binding request: staged checkpoint =
                               public trial file; report carries the staged sha; record written; a staged report with
                               another checkpoint sha gives no record
  W3.tree_writer_pair          real Step 5 tree writers (xgboost, CPU): staged report bytes = public family report =
                               canonical; staged checkpoint = public; sidecar step5_identity = the record
  W3.tree_writer_pair_lightgbm the same for lightgbm (real LGBMRegressor, staged save via booster_), and the pair
                               publishes (fix pass 1, 2026-11-27)
  W3.tree_writer_pair_catboost the same for catboost (real CatBoostRegressor, CPU) (fix pass 1, 2026-11-27)
  W3.interleaved_writers       Beta Q1: writer B overwrites the shared public checkpoint before writer A records its
                               pair: A's record names A's bytes
  W3.write_errors_step5        identity write fails: public checkpoint and sidecar as without binding, no record
  W3.compare_child_exit1       compare child writes a full pair then exits 1: not published; rc 0 children published;
                               compare still returns 0 with the same winner and restore
  W3.compare_publish_error     the publish step raising does not change the compare return, winner or restore
  W3.full_no_request           level full with no binding request: no --binding-* on any child, no staging dir
  W3.one_report_per_family     only the final NN path and the compare child command carry the request (source check)
  W3.runner_bound_then_unbound bound A then unbound B, and unbound B alone, with a usable live model present: no_checkpoint
                               (FlagUnbound), nothing attached, LL-8 never started
  W3.runner_ll8_handoff        bound flag: the runner finds the staged checkpoint by its explicit path (sha match) when
                               the flag dir's sidecar points at other bytes, and reaches the LL-8 command with it
  W3.absent_thc                no id: health results and flag lines == 14815a7 (legacy, fixed, promoted, pointer-stale)
  W3.absent_step5              no id: compare child argv, summary, archive and restore == 14815a7; save_model sidecar and
                               checkpoint bytes == 14815a7; NN worker report keys == 14815a7, no staging dir
  W3.installed_watcher         POST-LANDING: the INSTALLED watcher and manifest: one id per dispatch on argv and in the
                               health call, a retry gets a new id, a caller id is overwritten, a freshness skip makes no
                               id, a stale result of this run does not reset the registry; with the 14815a7 manifest the
                               argv and health call == 14815a7. SKIP with the reason when the owner patch is not
                               installed. Negative control inside the gate (health call without the id must fail).
Fix pass 2 (2026-11-27, Beta X-W3-BINDING R2):
  W3.nn_fallback_no_publish    Q2: the NN worker exits 0 with a complete staged pair but its metrics JSON lacks the
                               required keys; inline fallback allowed: the real single-shot path completes, no pending
                               record is written, nothing is published (also with a leftover pending record of the
                               rejected worker, and with a pending record this process did not write); the later flag
                               is unbound. Positive control: an accepted worker's pair publishes and binds
  W3.fixed_winner_reread       Q3: the public winner report is replaced after the graded read and read again: the
                               flag carries the graded bytes' sha256 and binds to the graded run's pair
  W3.pointer_swap              Q5: the compare pointer is replaced right after the read: the checked pointer routes
  W3.strict_stale_grade_policy Q5: stale_report_action 'grade' with WATCHER's id: a stale report is not graded in either
                               grader (PROCEED_WITH_NOTE / stale, no count, no flag); with no id it is graded
  W3.absent_thc (extended)     Q4, Q5: invalid UTF-8 report (legacy, fixed canonical, fixed promoted family) and the
                               'grade' policy with no id: == 14815a7
"""
import contextlib
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
BASE_REV = "14815a7"
RID = "s5-20261127T120000Z-aaaa1111"
RID_B = "s5-20261127T120000Z-bbbb2222"
STARTED = "2026-11-27T12:00:00+00:00"
PY = sys.executable


# ── helpers ─────────────────────────────────────────────────────────────────

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2))
    return p


def load_as(name, path):
    s = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def base_file(rel, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(subprocess.run(["git", "-C", str(REPO), "show", f"{BASE_REV}:{rel}"], check=True,
                                    capture_output=True).stdout)
    return dest


_MODS = {}


def mods(code=None):
    """Modules under test: the tree's files, or (mutants) a dict of replacement paths."""
    code = code or {}
    key = tuple(sorted(code.items()))
    if key not in _MODS:
        paths = {"td": REPO / "training_diagnostics.py", "thc": REPO / "training_health_check.py",
                 "wrr": REPO / "scripts" / "diagnostics" / "why_report_runner.py",
                 "mpo": REPO / "meta_prediction_optimizer_anti_overfit.py"}
        paths.update({k: Path(v) for k, v in code.items()})
        tag = hashlib.sha1(repr(key).encode()).hexdigest()[:8]
        td = load_as(f"td_{tag}", paths["td"])
        with td_as(td):
            m = {"td": td, "paths": paths, "thc": load_as(f"thc_{tag}", paths["thc"]),
                 "wrr": load_as(f"wrr_{tag}", paths["wrr"])}
        m["mpo_path"] = paths["mpo"]
        _MODS[key] = m
    return _MODS[key]


@contextlib.contextmanager
def td_as(td):
    old = sys.modules.get("training_diagnostics")
    sys.modules["training_diagnostics"] = td
    try:
        yield
    finally:
        if old is not None:
            sys.modules["training_diagnostics"] = old
        else:
            sys.modules.pop("training_diagnostics", None)


@contextlib.contextmanager
def cwd(d):
    old = os.getcwd()
    Path(d).mkdir(parents=True, exist_ok=True)
    os.chdir(d)
    try:
        yield Path(d)
    finally:
        os.chdir(old)


def critical_report(mt, **extra):
    return dict({"model_type": mt, "status": "complete",
                 "training_summary": {"rounds_captured": 17, "final_train_loss": 0.9, "final_val_loss": 0.95,
                                      "best_val_round": 14},
                 "diagnosis": {"severity": "critical", "issues": ["synthetic critical"], "suggested_fixes": []}},
                **extra)


def fixed_report(mt, critical, **extra):
    best, vl = (1, 2.0) if critical else (14, 1.05)
    return dict({"model_type": mt, "status": "complete",
                 "training_summary": {"best_val_round": best, "rounds_captured": 17, "final_train_loss": 1.0,
                                      "final_val_loss": vl},
                 "diagnosis": {"severity": "critical" if critical else "ok", "issues": [f"{mt}-ISSUE"],
                               "suggested_fixes": [f"{mt}-FIX"]}}, **extra)


def policies(root, **td):
    td.setdefault("model_skip_rules", {"max_consecutive_critical": 2})
    write_json(Path(root) / "watcher_policies.json", {"training_diagnostics": td})


def stage_pair(td, root, rid, mid, fam, report, ck_bytes=b"model bytes", publish=True, public=("family", "canonical"),
               ext=".pth"):
    """A final-model writer, as built: staged checkpoint first, then the report bytes (carrying the run id, model id
    and, for the NN, the staged checkpoint's sha) to staging and the public paths; pending record; publish."""
    req = td.w3_request(rid, STARTED, mid, fam, root=str(Path(root) / "diagnostics_outputs"))
    st = Path(req["stage_dir"])
    st.mkdir(parents=True, exist_ok=True)
    (st / f"best_model{ext}").write_bytes(ck_bytes)
    rep = dict(report, model_type=fam, diagnostics_run_id=rid, diagnostics_run_started_at=STARTED,
               binding_model_id=mid)
    if fam == "neural_net":
        rep["checkpoint_sha256"] = hashlib.sha256(ck_bytes).hexdigest()
    data = json.dumps(rep, indent=2)
    (st / f"{fam}_diagnostics.json").write_text(data)
    dd = Path(root) / "diagnostics_outputs"
    if "family" in public:
        (dd / f"{fam}_diagnostics.json").write_text(data)
    if "canonical" in public:
        (dd / "training_diagnostics.json").write_text(data)
    rec = td.w3_write_pending(req, fam, f"best_model{ext}", f"{fam}_diagnostics.json", fam == "neural_net")
    if publish:
        ok, why = td.w3_publish(str(st), rid, fam)
        assert ok, why
    return req, rec


def check(m, root, n=1, **kw):
    """Run the health check n times in root; return (results, flag lines)."""
    with cwd(root), td_as(m["td"]):
        res = [m["thc"].check_training_health(**kw) for _ in range(n)]
        fp = Path("diagnostics_outputs") / "why_report_flags.jsonl"
        lines = [json.loads(l) for l in fp.read_text().splitlines()] if fp.exists() else []
    return res, lines


def preset_registry(root, fam, n=1):
    write_json(Path(root) / "diagnostics_outputs" / "model_skip_registry.json",
               {fam: {"consecutive_critical": n, "last_critical": "2026-11-27T11:00:00+00:00",
                      "critical_since": "2026-11-27T10:00:00+00:00"}})


# ── health-check gates ──────────────────────────────────────────────────────

def g_bound_nn_legacy(m, d):
    root = d / "r"
    policies(root)
    preset_registry(root, "neural_net")
    req, rec = stage_pair(m["td"], root, RID, "nn-1", "neural_net", critical_report("neural_net"))
    res, lines = check(m, root, binding_run_id=RID, run_started_at=STARTED, models_dir="models/elsewhere")
    ln = lines[-1] if lines else {}
    st = (root / "diagnostics_outputs" / RID / "nn-1").resolve()
    ok = (len(lines) == 1 and ln.get("binding") == "bound" and ln.get("checkpoint_path") == str(st / "best_model.pth")
          and ln.get("models_dir") == str(st) and ln.get("checkpoint_sha256") == sha(st / "best_model.pth")
          and ln.get("report_sha256") == sha(root / "diagnostics_outputs" / "training_diagnostics.json")
          and ln.get("step5_run_id") == RID and ln.get("flagged_model_error") is None
          and res[0]["why_report_needed"]["binding"] == "bound" and res[0]["binding_run_id"] == RID
          and res[0]["action"] == "RETRY")
    return ok, f"lines {len(lines)}; line {ln.get('binding')} {ln.get('binding_reason')}"


def g_strict_stale(m, d):
    out = []
    for tag, rep in (("other_id", critical_report("neural_net", diagnostics_run_id=RID_B)),
                     ("no_id", critical_report("neural_net"))):
        root = d / tag
        policies(root)
        preset_registry(root, "neural_net")
        write_json(root / "diagnostics_outputs" / "training_diagnostics.json", rep)   # written after the start
        res, lines = check(m, root, binding_run_id=RID, run_started_at="2000-01-01T00:00:00+00:00")
        reg = json.loads((root / "diagnostics_outputs" / "model_skip_registry.json").read_text())
        out.append((tag, res[0]["severity"], len(lines), reg["neural_net"]["consecutive_critical"]))
    return all(o[1] == "stale" and o[2] == 0 and o[3] == 1 for o in out), str(out)


def _one_line_unbound(m, d, prep, fam="neural_net", want=None):
    root = d / "r"
    policies(root)
    preset_registry(root, fam)
    prep(root)
    res, lines = check(m, root, binding_run_id=RID, run_started_at=STARTED)
    ln = lines[-1] if lines else {}
    ok = (len(lines) == 1 and ln.get("binding") == "unbound" and ln.get("checkpoint_sha256") is None
          and ln.get("checkpoint_path") is None and str(ln.get("flagged_model_error", "")).startswith("unbound:")
          and (want is None or want in str(ln.get("binding_reason"))))
    return ok, f"lines {len(lines)}; {ln.get('binding')}: {ln.get('binding_reason')}"


def g_report_sha_link(m, d):
    def prep(root):
        stage_pair(m["td"], root, RID, "nn-1", "neural_net", critical_report("neural_net"))
        # a trial report stamped with the same run id and model id replaces the public one before grading
        rep = json.loads((root / "diagnostics_outputs" / "training_diagnostics.json").read_text())
        rep["training_summary"]["rounds_captured"] = 3
        write_json(root / "diagnostics_outputs" / "training_diagnostics.json", rep)
    return _one_line_unbound(m, d, prep, want="no published identity record")


def g_staged_ck_changed(m, d):
    def prep(root):
        stage_pair(m["td"], root, RID, "nn-1", "neural_net", critical_report("neural_net"))
        (root / "diagnostics_outputs" / RID / "nn-1" / "best_model.pth").write_bytes(b"other bytes")
    return _one_line_unbound(m, d, prep)


def g_nn_report_ck(m, d):
    def prep(root):
        req, rec = stage_pair(m["td"], root, RID, "nn-1", "neural_net", critical_report("neural_net"))
        st = root / "diagnostics_outputs" / RID / "nn-1"
        (st / "best_model.pth").write_bytes(b"other model")      # a forged record, consistent with its own file
        idp = st / "identity.json"
        r = json.loads(idp.read_text())
        r["checkpoint_sha256"] = sha(st / "best_model.pth")
        idp.write_text(json.dumps(r))
    return _one_line_unbound(m, d, prep, want="checkpoint_sha256")


def g_override_family(m, d):
    def prep(root):
        stage_pair(m["td"], root, RID, "nn-1", "neural_net", critical_report("neural_net"))
        write_json(root / "diagnostics_outputs" / "compare_models_summary.json", {"winner_model_type": "catboost"})
    return _one_line_unbound(m, d, prep, fam="catboost", want="own family")


def g_unbound_tree_legacy(m, d):
    def prep(root):   # catboost: no tree report at level legacy; the canonical report is an old NN one of this id
        stage_pair(m["td"], root, RID, "nn-1", "neural_net", critical_report("neural_net"))
        write_json(root / "diagnostics_outputs" / "training_diagnostics.json",
                   critical_report("catboost", diagnostics_run_id=RID))
    return _one_line_unbound(m, d, prep, fam="catboost", want="no tree report at level legacy")


def g_fixed_promoted(m, d):
    root = d / "r"
    policies(root, grading_rules="fixed", grade_scope="all_families")
    preset_registry(root, "neural_net")
    stage_pair(m["td"], root, RID, "cmp-catboost", "catboost", fixed_report("catboost", False), b"cb", ext=".cbm",
               public=("family",))
    stage_pair(m["td"], root, RID, "cmp-neural_net", "neural_net", fixed_report("neural_net", True), b"nn")
    write_json(root / "diagnostics_outputs" / "compare_models_summary_latest.json",
               {"run_id": "S88_x", "diagnostics_run_id": RID, "models": {"catboost": {}, "neural_net": {}},
                "winner_best_effort": {"model_type": "catboost", "score": 0.9}})
    res, lines = check(m, root, binding_run_id=RID, run_started_at=STARTED)
    ln = lines[-1] if lines else {}
    st = (root / "diagnostics_outputs" / RID / "cmp-neural_net").resolve()
    ok = (res[0].get("grade_set_by") == "neural_net" and len(lines) == 1 and ln.get("model_type") == "neural_net"
          and ln.get("binding") == "bound" and ln.get("checkpoint_path") == str(st / "best_model.pth")
          and ln.get("report_sha256") == sha(root / "diagnostics_outputs" / "neural_net_diagnostics.json"))
    return ok, f"grade_set_by {res[0].get('grade_set_by')}; {ln.get('binding')}: {ln.get('binding_reason')}"


def g_fixed_leftover_pointer(m, d):
    root = d / "r"
    policies(root, grading_rules="fixed")
    write_json(root / "diagnostics_outputs" / "training_diagnostics.json", fixed_report("xgboost", False,
                                                                                       diagnostics_run_id=RID))
    write_json(root / "diagnostics_outputs" / "catboost_diagnostics.json",
               fixed_report("catboost", False, diagnostics_run_id=RID_B))
    write_json(root / "diagnostics_outputs" / "compare_models_summary_latest.json",
               {"run_id": "S88_old", "diagnostics_run_id": RID_B, "models": {"catboost": {}},
                "winner_best_effort": {"model_type": "catboost", "score": 0.9}})
    res, _ = check(m, root, binding_run_id=RID, run_started_at=STARTED)
    r = res[0]
    ok = (r["severity"] == "ok" and r["model_type"] == "xgboost" and r["report_source"] == "canonical"
          and "pointer ignored" in str(r.get("pointer_ignored")))
    return ok, f"severity {r['severity']} model {r['model_type']} source {r.get('report_source')}"


def g_report_swapped(m, d):
    root = d / "r"
    policies(root)
    preset_registry(root, "neural_net")
    stage_pair(m["td"], root, RID, "nn-1", "neural_net", critical_report("neural_net"))
    thc = m["thc"]
    orig = thc._severity_to_action

    def swap(*a, **k):   # another process rewrites the canonical report after it was graded
        write_json(Path("diagnostics_outputs") / "training_diagnostics.json", critical_report("neural_net"))
        return orig(*a, **k)
    thc._severity_to_action = swap
    try:
        res, lines = check(m, root, binding_run_id=RID, run_started_at=STARTED)
    finally:
        thc._severity_to_action = orig
    ln = lines[-1] if lines else {}
    return ln.get("binding") == "bound", f"{ln.get('binding')}: {ln.get('binding_reason')}"


def g_read_write_errors(m, d):
    out = []
    for tag in ("identity_unreadable", "checkpoint_unreadable"):
        root = d / tag
        policies(root)
        preset_registry(root, "neural_net")
        stage_pair(m["td"], root, RID, "nn-1", "neural_net", critical_report("neural_net"))
        st = root / "diagnostics_outputs" / RID / "nn-1"
        if tag == "identity_unreadable":
            (st / "identity.json").write_text("{not json")
        else:
            (st / "best_model.pth").unlink()
            (st / "best_model.pth").mkdir()
        res, lines = check(m, root, binding_run_id=RID, run_started_at=STARTED)
        out.append((tag, len(lines), lines[-1].get("binding") if lines else None, res[0]["action"],
                    (lines[-1].get("binding_reason") or "")[:60] if lines else ""))
    return all(o[1] == 1 and o[2] == "unbound" and o[3] == "RETRY" for o in out), str(out)


def g_deadline(m, d):
    thc = m["thc"]
    root = d / "r"
    policies(root, why_flag_binding_deadline_seconds=0.5)
    preset_registry(root, "neural_net")
    stage_pair(m["td"], root, RID, "nn-1", "neural_net", critical_report("neural_net"))
    orig = thc._w3_bind_io

    def slow(*a, **k):
        time.sleep(3)
        return orig(*a, **k)
    thc._w3_bind_io = slow
    t0 = time.monotonic()
    try:
        res, lines = check(m, root, binding_run_id=RID, run_started_at=STARTED)
    finally:
        thc._w3_bind_io = orig
    dt = time.monotonic() - t0
    ln = lines[-1] if lines else {}
    bad = [thc._w3_deadline({"why_flag_binding_deadline_seconds": v}) for v in ("x", -1, 0, float("nan"), True)]
    ok = (dt < 2.0 and len(lines) == 1 and ln.get("binding") == "unbound" and "deadline" in ln.get("binding_reason", "")
          and all(b[0] == 10.0 and b[1] for b in bad) and thc._w3_deadline({}) == (10.0, None)
          and res[0]["action"] == "RETRY")
    return ok, f"{dt:.2f}s; lines {len(lines)}; {ln.get('binding_reason')}; bad settings {bad[:2]}"


def g_publish_recheck(m, d):
    td = m["td"]
    root = d / "r"
    req, rec = stage_pair(td, root, RID, "nn-1", "neural_net", critical_report("neural_net"), publish=False)
    st = Path(req["stage_dir"])
    (st / "best_model.pth").write_bytes(b"changed")
    ok_pub, why = td.w3_publish(str(st), RID, "neural_net")
    return (not ok_pub and not (st / "identity.json").exists() and rec is not None), why


# ── producer gates (real writers) ───────────────────────────────────────────

NN_PARAMS = json.dumps({"epochs": 2, "hidden_layers": [8], "learning_rate": 0.01, "batch_size": 32})


def tiny_npz(p, n=120, f=6, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, f)).astype("float32")
    y = (X[:, 0] * 0.5 + rng.normal(size=n) * 0.1).astype("float32")
    k = int(n * 0.8)
    np.savez(p, X_train=X[:k], y_train=y[:k], X_val=X[k:], y_val=y[k:])
    return p


def run_tst(tst_path, root, extra=()):
    root = Path(root)
    (root / "models").mkdir(parents=True, exist_ok=True)
    npz = tiny_npz(root / "split.npz")
    cmd = [PY, "-B", str(tst_path), "--model-type", "neural_net", "--data-path", str(npz), "--save-model",
           "--model-output-dir", str(root / "models"), "--params", NN_PARAMS, "--enable-diagnostics",
           "--normalize-features", "--use-leaky-relu", *extra]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=f"{REPO}{os.pathsep}{os.environ.get('PYTHONPATH', '')}")
    p = subprocess.run(cmd, cwd=str(root), env=env, capture_output=True, text=True, timeout=600)
    return p


def g_nn_writer_pair(m, d):
    td = m["td"]
    root = d / "nn"
    p = run_tst(REPO / "train_single_trial.py", root,
                ["--binding-run-id", RID, "--binding-run-started-at", STARTED, "--binding-model-id", "nn-1"])
    st = root / "diagnostics_outputs" / RID / "nn-1"
    pub = sorted((root / "models").glob("*.pth"))
    if p.returncode != 0 or not pub:
        return False, f"worker rc {p.returncode}: {p.stderr[-600:]}"
    rep = json.loads((st / "neural_net_diagnostics.json").read_text())
    same_bytes = ((st / "neural_net_diagnostics.json").read_bytes()
                  == (root / "diagnostics_outputs" / "neural_net_diagnostics.json").read_bytes()
                  == (root / "diagnostics_outputs" / "training_diagnostics.json").read_bytes())
    with cwd(root):
        req = td.w3_request(RID, STARTED, "nn-1", "neural_net")
        rec = td.w3_write_pending(req, "neural_net", "best_model.pth", "neural_net_diagnostics.json", True)
        # negative: a staged report naming another checkpoint gives no record
        bad = dict(rep, checkpoint_sha256="0" * 64)
        (st / "neural_net_diagnostics.json").write_text(json.dumps(bad))
        rec_bad = td.w3_write_pending(req, "neural_net", "best_model.pth", "neural_net_diagnostics.json", True)
    ok = (sha(st / "best_model.pth") == sha(pub[0]) == rep.get("checkpoint_sha256") and same_bytes
          and rep.get("diagnostics_run_id") == RID and rep.get("binding_model_id") == "nn-1"
          and rec is not None and rec["checkpoint_sha256"] == sha(pub[0]) and rec_bad is None)
    return ok, f"staged == public {sha(st / 'best_model.pth') == sha(pub[0])}; report bytes same {same_bytes}; " \
               f"record {bool(rec)}; bad record {rec_bad is not None}"


def _mpo(m, tag):
    import logging
    with td_as(m["td"]):
        mpo = load_as(f"mpo_{tag}_{hashlib.sha1(str(m['mpo_path']).encode()).hexdigest()[:6]}", m["mpo_path"])
    return mpo, logging.getLogger("w3test")


def _xgb(seed=0, n=150):
    import numpy as np
    import xgboost as xgb
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = X[:, 0] + rng.normal(size=n) * 0.1
    mdl = xgb.XGBRegressor(n_estimators=5 + seed, max_depth=2, n_jobs=2, tree_method="hist")
    mdl.fit(X[:120], y[:120], eval_set=[(X[120:], y[120:])], verbose=False)
    return mdl, X, y


SAVE_KW = dict(feature_schema={"feature_names": [f"f{i}" for i in range(5)]}, training_metrics={"r2": 0.5},
               signal_quality={"prediction_allowed": True}, data_context={}, hyperparameters={},
               provenance={"run_id": "step5_x", "started_at": "t"})


def _tree_report_and_save(mpo, lg, m, mdl, X, y, req, out_dir, fam="xgboost"):
    opt = object.__new__(mpo.AntiOverfitMetaOptimizer)
    opt.__dict__.update(model_type=fam, _cli_enable_diagnostics=True, _cli_diagnostics_level="full",
                        feature_schema={"feature_names": [f"f{i}" for i in range(5)]}, logger=lg, _w3_req=req,
                        _cli_diagnostics_options={"level": "full", "run_id": req["run_id"] if req else None})
    with td_as(m["td"]):
        opt._ch14_emit_inline_tree_report({"model": mdl, "metrics": {"r2": 0.5, "val_mse": 0.1}},
                                          X[:120], y[:120], X[120:], y[120:])
        tr = mpo.MultiModelTrainer(device="cpu", logger=lg)
        tr.save_model(model=mdl, model_type=fam, output_dir=str(out_dir), **SAVE_KW,
                      **({"w3_req": req} if req and opt._w3_tree_report_ok else {}))
    return opt


def g_tree_writer_pair(m, d):
    mpo, lg = _mpo(m, "tree")
    mdl, X, y = _xgb()
    with cwd(d / "t") as root:
        req = m["td"].w3_request(RID, STARTED, "xgb-1", "xgboost")
        _tree_report_and_save(mpo, lg, m, mdl, X, y, req, root / "models")
        st = Path(req["stage_dir"]).resolve()
        dd = root / "diagnostics_outputs"
        side = json.loads((root / "models" / "best_model.meta.json").read_text())
        pend = json.loads((st / "identity.pending.json").read_text())
    ok = ((st / "xgboost_diagnostics.json").read_bytes() == (dd / "xgboost_diagnostics.json").read_bytes()
          == (dd / "training_diagnostics.json").read_bytes()
          and sha(st / "best_model.json") == sha(root / "models" / "best_model.json")
          and side.get("step5_identity", {}).get("checkpoint_sha256") == sha(st / "best_model.json")
          and side["step5_identity"]["report_sha256"] == sha(dd / "training_diagnostics.json")
          and pend["checkpoint_sha256"] == side["step5_identity"]["checkpoint_sha256"]
          and json.loads((dd / "training_diagnostics.json").read_text()).get("diagnostics_run_id") == RID
          and side["agent_metadata"]["run_id"] == "step5_x")
    return ok, f"sidecar block {side.get('step5_identity', {}).get('model_id')}"


def _tree_model(fam, seed=0, n=150):
    """[W3 fix pass 1] a small real fitted model of one tree family (CPU), the sklearn API Step 5 uses."""
    import numpy as np
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = X[:, 0] + rng.normal(size=n) * 0.1
    if fam == "lightgbm":
        import lightgbm as lgb
        mdl = lgb.LGBMRegressor(n_estimators=8, num_leaves=4, n_jobs=2, verbose=-1, device="cpu")
        mdl.fit(X[:120], y[:120], eval_set=[(X[120:], y[120:])])
    else:
        import catboost as cb
        mdl = cb.CatBoostRegressor(iterations=8, depth=2, thread_count=2, task_type="CPU", verbose=False,
                                   allow_writing_files=False)
        mdl.fit(X[:120], y[:120], eval_set=(X[120:], y[120:]))
    return mdl, X, y


def _g_tree_writer_pair_fam(m, d, fam):
    """[W3 fix pass 1] W3.tree_writer_pair for lightgbm / catboost: the real staged save (booster_ for lightgbm)."""
    mpo, lg = _mpo(m, f"tree_{fam}")
    mdl, X, y = _tree_model(fam)
    ext = {"lightgbm": ".txt", "catboost": ".cbm"}[fam]
    with cwd(d / "t") as root:
        req = m["td"].w3_request(RID, STARTED, f"{fam}-1", fam)
        _tree_report_and_save(mpo, lg, m, mdl, X, y, req, root / "models", fam=fam)
        st = Path(req["stage_dir"]).resolve()
        dd = root / "diagnostics_outputs"
        side = json.loads((root / "models" / "best_model.meta.json").read_text())
        pend = json.loads((st / "identity.pending.json").read_text()) if (st / "identity.pending.json").exists() else {}
        ok_pub, why = m["td"].w3_publish(str(st), RID, fam)
    blk = side.get("step5_identity") or {}
    ok = ((st / f"{fam}_diagnostics.json").read_bytes() == (dd / f"{fam}_diagnostics.json").read_bytes()
          == (dd / "training_diagnostics.json").read_bytes()
          and (st / f"best_model{ext}").is_file() and (st / f"best_model{ext}").stat().st_size > 0
          and sha(st / f"best_model{ext}") == sha(root / "models" / f"best_model{ext}")
          and blk.get("checkpoint_sha256") == sha(st / f"best_model{ext}") and blk.get("model_type") == fam
          and blk.get("checkpoint_file") == f"best_model{ext}"
          and blk.get("report_sha256") == sha(dd / "training_diagnostics.json")
          and pend.get("checkpoint_sha256") == blk.get("checkpoint_sha256")
          and side.get("model_type") == fam and ok_pub and (st / "identity.json").is_file())
    return ok, f"sidecar block {blk.get('model_id')} {blk.get('checkpoint_file')}; publish {ok_pub} {why}"


def g_tree_writer_pair_lightgbm(m, d):
    return _g_tree_writer_pair_fam(m, d, "lightgbm")


def g_tree_writer_pair_catboost(m, d):
    return _g_tree_writer_pair_fam(m, d, "catboost")


def g_interleaved_writers(m, d):
    """Beta Q1: B rewrites the shared public checkpoint after A saved and before A records its pair."""
    mpo, lg = _mpo(m, "inter")
    a_mdl, X, y = _xgb(0)
    b_mdl, _, _ = _xgb(3)
    with cwd(d / "i") as root:
        out = root / "models"
        a_req = m["td"].w3_request(RID, STARTED, "xgb-a", "xgboost")
        b_req = m["td"].w3_request(RID_B, STARTED, "xgb-b", "xgboost")
        real_copy = shutil.copyfile
        state = {"done": False}

        def copy_hook(src, dst, *a, **k):
            if not state["done"] and (str(src).endswith("best_model.json") or str(dst).endswith("best_model.json")):
                state["done"] = True
                with td_as(m["td"]):          # writer B, start to finish, on the shared public path
                    shutil.copyfile = real_copy
                    _tree_report_and_save(mpo, lg, m, b_mdl, X, y, b_req, out)
                    shutil.copyfile = copy_hook
            return real_copy(src, dst, *a, **k)
        shutil.copyfile = copy_hook
        try:
            _tree_report_and_save(mpo, lg, m, a_mdl, X, y, a_req, out)
        finally:
            shutil.copyfile = real_copy
        a_bytes = root / "a_ref.json"
        a_mdl.save_model(str(a_bytes))
        a_pend = json.loads((Path(a_req["stage_dir"]) / "identity.pending.json").read_text())
    ok = state["done"] and a_pend["checkpoint_sha256"] == sha(a_bytes) and a_pend["run_id"] == RID
    return ok, f"interleaved {state['done']}; A record names A's bytes {a_pend['checkpoint_sha256'] == sha(a_bytes)}"


def g_write_errors_step5(m, d):
    mpo, lg = _mpo(m, "werr")
    mdl, X, y = _xgb()
    td = m["td"]
    orig = td.w3_atomic_write_json

    def boom(*a, **k):
        raise OSError("injected write error")
    with cwd(d / "w") as root:
        req = td.w3_request(RID, STARTED, "xgb-1", "xgboost")
        td.w3_atomic_write_json = boom
        try:
            _tree_report_and_save(mpo, lg, m, mdl, X, y, req, root / "models")
        finally:
            td.w3_atomic_write_json = orig
        side = json.loads((root / "models" / "best_model.meta.json").read_text())
        ref = root / "ref.json"
        mdl.save_model(str(ref))
        ok = ("step5_identity" not in side and sha(root / "models" / "best_model.json") == sha(ref)
              and not (Path(req["stage_dir"]) / "identity.pending.json").exists())
    return ok, f"sidecar keys {sorted(side)[:4]}...; no record"


# ── compare (stub children) ─────────────────────────────────────────────────

STUB = r'''
import json, os, sys, hashlib
sys.path.insert(0, os.environ["W3_REPO"])
import training_diagnostics as td
a = sys.argv[1:]
with open(os.environ["W3_ARGV_LOG"], "a") as f:
    f.write(json.dumps(a) + "\n")
def opt(k):
    return a[a.index(k) + 1] if k in a else None
fam = opt("--model-type")
ext = {"neural_net": ".pth", "xgboost": ".json", "lightgbm": ".txt", "catboost": ".cbm"}[fam]
score = {"neural_net": 0.1, "lightgbm": 0.2, "xgboost": 0.3, "catboost": 0.9}[fam]
ckb = f"{fam} model bytes".encode()
os.makedirs("models/reinforcement", exist_ok=True)
os.makedirs("diagnostics_outputs", exist_ok=True)
open(f"models/reinforcement/best_model{ext}", "wb").write(ckb)
json.dump({"model_type": fam, "checkpoint_path": f"models/reinforcement/best_model{ext}",
           "training_metrics": {"r2": score}}, open("models/reinforcement/best_model.meta.json", "w"))
rep = {"model_type": fam, "diagnosis": {"severity": "critical" if fam == "neural_net" else "ok"}}
rid = opt("--binding-run-id")
if rid:
    req = td.w3_request(rid, opt("--binding-run-started-at"), opt("--binding-model-id"), fam)
    os.makedirs(req["stage_dir"], exist_ok=True)
    open(os.path.join(req["stage_dir"], f"best_model{ext}"), "wb").write(ckb)
    rep.update(diagnostics_run_id=rid, binding_model_id=req["model_id"])
    if fam == "neural_net":
        rep["checkpoint_sha256"] = hashlib.sha256(ckb).hexdigest()
    open(os.path.join(req["stage_dir"], f"{fam}_diagnostics.json"), "w").write(json.dumps(rep, indent=2))
    td.w3_write_pending(req, fam, f"best_model{ext}", f"{fam}_diagnostics.json", fam == "neural_net")
if fam == "neural_net":
    open("diagnostics_outputs/training_diagnostics.json", "w").write(json.dumps(rep, indent=2))
sys.exit(1 if fam in os.environ.get("W3_FAIL", "").split(",") else 0)
'''

COMPARE_DRIVER = r'''
import json, os, sys, importlib.util as u
mpo_path, td_path, stub, mode = sys.argv[1:5]
sys.path.insert(0, os.environ["W3_REPO"])
s = u.spec_from_file_location("training_diagnostics", td_path)
td = u.module_from_spec(s); sys.modules["training_diagnostics"] = td; s.loader.exec_module(td)
if os.environ.get("W3_PUBLISH_RAISES"):
    def boom(*a, **k):
        raise OSError("injected publish error")
    td.w3_publish = boom
s = u.spec_from_file_location("mpo_drv", mpo_path)
m = u.module_from_spec(s); s.loader.exec_module(m)
sys.argv = [stub] + json.loads(os.environ.get("W3_MAIN_ARGV", "[]"))
if mode == "main":
    try:
        rc = m.main()
    except SystemExit as e:
        rc = e.code
else:
    rc = m._s88_run_compare_models(json.loads(os.environ["W3_ARGS"]))
print("RC", rc)
'''


def run_compare(d, mpo_path, td_path=REPO / "training_diagnostics.py", rid=None, fail="", mode="direct",
                main_argv=None, publish_raises=False, level="legacy"):
    root = Path(d)
    root.mkdir(parents=True, exist_ok=True)
    (root / "stub.py").write_text(STUB)
    (root / "driver.py").write_text(COMPARE_DRIVER)
    args = {"survivors": "s.json", "lottery_data": "l.json", "trials": 1, "enable_diagnostics": True,
            "diagnostics_level": level}
    if rid:
        args.update(binding_run_id=rid, binding_run_started_at=STARTED)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", W3_REPO=str(REPO),
               W3_ARGV_LOG=str(root / "argv.jsonl"), W3_FAIL=fail, W3_ARGS=json.dumps(args),
               W3_MAIN_ARGV=json.dumps(main_argv or []), PYTHONPATH=str(REPO))
    if publish_raises:
        env["W3_PUBLISH_RAISES"] = "1"
    p = subprocess.run([PY, "-B", str(root / "driver.py"), str(mpo_path), str(td_path), str(root / "stub.py"), mode],
                       cwd=str(root), env=env, capture_output=True, text=True, timeout=600)
    rc = re.findall(r"^RC (\S+)$", p.stdout, re.M)
    argv = [json.loads(l) for l in (root / "argv.jsonl").read_text().splitlines()] if (root / "argv.jsonl").exists() \
        else []
    summ = sorted((root / "diagnostics_outputs").glob("compare_models_summary_S88_*.json"))
    return {"rc": rc[-1] if rc else None, "stdout": p.stdout, "stderr": p.stderr[-1500:], "argv": argv,
            "summary": json.loads(summ[-1].read_text()) if summ else None, "root": root}


def g_compare_child_exit1(m, d):
    r = run_compare(d / "c", m["mpo_path"], m["paths"]["td"], rid=RID, fail="neural_net")
    dd = r["root"] / "diagnostics_outputs" / RID
    pub = {f: (dd / f"cmp-{f}" / "identity.json").exists() for f in ("neural_net", "lightgbm", "xgboost", "catboost")}
    pend_nn = (dd / "cmp-neural_net" / "identity.pending.json").exists()
    ok = (r["rc"] == "0" and pub == {"neural_net": False, "lightgbm": True, "xgboost": True, "catboost": True}
          and pend_nn and r["summary"]["winner_best_effort"]["model_type"] == "catboost"
          and (r["root"] / "models/reinforcement/best_model.cbm").read_bytes() == b"catboost model bytes")
    # and the health check: the NN (critical, canonical) is unbound
    if ok:
        policies(r["root"])
        preset_registry(r["root"], "neural_net")
        res, lines = check(m, r["root"], binding_run_id=RID, run_started_at=STARTED)
        ok = len(lines) == 1 and lines[0].get("binding") == "unbound"
    return ok, f"rc {r['rc']}; published {pub}; NN pending left {pend_nn}; {r['stderr'][-300:] if not ok else ''}"


def g_compare_publish_error(m, d):
    a = run_compare(d / "ok", m["mpo_path"], m["paths"]["td"], rid=RID)
    b = run_compare(d / "err", m["mpo_path"], m["paths"]["td"], rid=RID, publish_raises=True)
    same = (a["rc"] == b["rc"] == "0" and a["summary"]["winner_best_effort"] == b["summary"]["winner_best_effort"]
            and (a["root"] / "models/reinforcement/best_model.cbm").read_bytes()
            == (b["root"] / "models/reinforcement/best_model.cbm").read_bytes()
            and not list((b["root"] / "diagnostics_outputs" / RID).glob("*/identity.json")))
    return same, f"rc {a['rc']}/{b['rc']}; {b['stderr'][-300:] if not same else ''}"


def g_full_no_request(m, d):
    r = run_compare(d / "f", m["mpo_path"], m["paths"]["td"], mode="main",
                    main_argv=["--survivors", "s.json", "--lottery-data", "l.json", "--compare-models", "--trials", "1",
                               "--enable-diagnostics", "--diagnostics-level", "full"])
    has_req = any(any(x.startswith("--binding") for x in a) for a in r["argv"])
    has_diag_id = all("--diagnostics-run-id" in a for a in r["argv"])
    stage = [p for p in (r["root"] / "diagnostics_outputs").glob("s5-*")] if (r["root"] / "diagnostics_outputs").exists() \
        else []
    ok = r["rc"] == "0" and len(r["argv"]) == 4 and has_diag_id and not has_req and not stage
    return ok, f"rc {r['rc']}; children {len(r['argv'])}; diag id on all {has_diag_id}; binding flags {has_req}; " \
               f"staging dirs {len(stage)} {r['stderr'][-300:] if not ok else ''}"


def g_one_report_per_family(m, d):
    src = Path(m["mpo_path"]).read_text()
    tst = (REPO / "train_single_trial.py").read_text()
    lines = [i for i, l in enumerate(src.splitlines(), 1) if '"--binding-run-id"' in l]
    fold_fn = src[src.index("def _run_nn_optuna_trial"):src.index("def _run_nn_optuna_trial") + 6000]
    final_fn = src[src.index("def _run_nn_via_subprocess"):src.index("def _ch14_full")]
    ok = (len(lines) == 2 and '"--binding-run-id"' in final_fn and "--binding" not in fold_fn
          and "'binding': _w3_req" in tst and "args.model_type == 'neural_net'" in tst)
    return ok, f"--binding-run-id emitted at mpo lines {lines}"


# ── runner gates ────────────────────────────────────────────────────────────

def _runner_world(m, d, lines):
    wrr = m["wrr"]
    root = d / "root"
    (root / "diagnostics_outputs").mkdir(parents=True, exist_ok=True)
    (root / "diagnostics_outputs" / "why_report_flags.jsonl").write_text("".join(json.dumps(l) + "\n" for l in lines))
    out = Path(wrr.REPO) / "data" / "w3_runner_test" / f"{os.getpid()}_{time.time_ns()}"
    return root, out


def _flag_line(at, fam="neural_net", **extra):
    return dict({"schema": "ch14_why_report_flag/1", "flag_id": f"{fam}@2026-11-27T10:00:00+00:00",
                 "flagged_at": at, "model_type": fam, "critical_since": "2026-11-27T10:00:00+00:00",
                 "consecutive_critical": 3, "threshold": 3}, **extra)


def g_runner_bound_then_unbound(m, d):
    wrr = m["wrr"]
    ck = d / "a" / "best_model.pth"
    ck.parent.mkdir(parents=True)
    ck.write_bytes(b"A")
    live = d / "live"    # a usable live model of the family
    live.mkdir()
    (live / "best_model.pth").write_bytes(b"LIVE")
    write_json(live / "best_model.meta.json", {"model_type": "neural_net", "checkpoint_path": str(live / "best_model.pth")})
    bound = _flag_line("2026-11-27T11:00:00+00:00", binding_schema="w3/1", binding="bound", checkpoint_sha256=sha(ck),
                       checkpoint_path=str(ck), models_dir=str(ck.parent))
    unb = _flag_line("2026-11-27T12:00:00+00:00", binding_schema="w3/1", binding="unbound", checkpoint_sha256=None,
                     binding_reason="synthetic unbound", flagged_model_error="unbound: synthetic unbound")
    out = []
    for tag, lines in (("bound_then_unbound", [bound, unb]), ("unbound_alone", [unb])):
        root, outr = _runner_world(m, d / tag, lines)
        calls = []
        res = wrr.run_for_flag(lines[0]["flag_id"], [], root=str(root), out_root=str(outr), timeout_s=60,
                               models_dir=str(live), ll8_cmd=[PY, "-c", "import sys; sys.exit(0)"],
                               votes_builder=lambda *a: calls.append(a))
        att = root / "diagnostics_outputs" / "why_report_attachments.jsonl"
        out.append((tag, res.get("status"), res.get("reason"), res.get("error_class"), "ll8" in res.get("children", {}),
                    att.exists() and att.read_text().strip() != "", bool(calls)))
    ok = all(o[1] == "failed" and o[2] == "no_checkpoint" and o[3] == "FlagUnbound" and not o[4] and not o[5]
             and not o[6] for o in out)
    return ok, str(out)


def g_runner_ll8_handoff(m, d):
    wrr = m["wrr"]
    st = d / "stage"
    st.mkdir(parents=True)
    (st / "best_model.pth").write_bytes(b"flagged model")
    canon = d / "canonical"
    canon.mkdir()
    (canon / "best_model.pth").write_bytes(b"other model")
    write_json(st / "best_model.meta.json", {"model_type": "neural_net", "checkpoint_path": str(canon / "best_model.pth")})
    want = sha(st / "best_model.pth")
    line = _flag_line("2026-11-27T11:00:00+00:00", binding_schema="w3/1", binding="bound", checkpoint_sha256=want,
                      checkpoint_path=str(st / "best_model.pth"), models_dir=str(st))
    root, outr = _runner_world(m, d, [line])
    seen = {}
    saved = {k: getattr(wrr, k) for k in ("find_block", "votes_source", "wf_frame", "check_inputs", "build_spec")}

    def fb(family, ck_sha, roots):
        seen["find_block_sha"] = ck_sha
        return {"summary_path": "s", "arm": {"arm_id": "a1"}, "summary": {"block": {"block": 1, "start_index": 0,
                                                                                     "end_index": 9}}}

    def bs(flag_id, family, ck, cand, out, analysis):
        seen["ck"] = ck
        p = Path(out) / "spec.json"
        p.write_text("{}")
        return p
    wrr.__dict__.update(find_block=fb, votes_source=lambda c: "stub", wf_frame=lambda c: None,
                        check_inputs=lambda c: None, build_spec=bs)
    rec = d / "ll8_argv.json"
    try:
        res = wrr.run_for_flag(line["flag_id"], [], root=str(root), out_root=str(outr), timeout_s=60,
                               votes_builder=lambda *a: "v.npz",
                               ll8_cmd=[PY, "-c", f"import json,sys; json.dump(sys.argv, open({str(rec)!r}, 'w'))"])
    finally:
        wrr.__dict__.update(saved)
    ck = seen.get("ck") or {}
    argv = json.loads(rec.read_text()) if rec.exists() else []
    ok = (ck.get("checkpoint_path") == str(st / "best_model.pth") and ck.get("found_in") == "flag_checkpoint_path"
          and ck.get("checkpoint_sha256") == want and seen.get("find_block_sha") == want
          and "--model-family" in argv and "neural_net" in argv)
    return ok, f"found_in {ck.get('found_in')}; LL-8 started {bool(argv)}; status {res.get('status')} {res.get('message', '')[:200]}"


# ── absent-is-today gates ───────────────────────────────────────────────────

TS = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?(\+00:00|Z)?")


def norm(x):
    return json.loads(TS.sub("<ts>", json.dumps(x, sort_keys=True, default=str)))


def g_absent_thc(m, d):
    base = mods({"thc": base_file("training_health_check.py", d / "base" / "training_health_check.py")})
    cases = []

    def mk(tag, setup, kw):
        cases.append((tag, setup, kw))

    def legacy(root):
        policies(root)
        preset_registry(root, "neural_net")
        write_json(root / "diagnostics_outputs" / "training_diagnostics.json", critical_report("neural_net"))

    def fixed_prom(root):
        policies(root, grading_rules="fixed", grade_scope="all_families")
        preset_registry(root, "neural_net")
        write_json(root / "diagnostics_outputs" / "catboost_diagnostics.json", fixed_report("catboost", False))
        write_json(root / "diagnostics_outputs" / "neural_net_diagnostics.json", fixed_report("neural_net", True))
        write_json(root / "diagnostics_outputs" / "compare_models_summary_latest.json",
                   {"run_id": "S88", "models": {"catboost": {}, "neural_net": {}},
                    "winner_best_effort": {"model_type": "catboost", "score": 0.9}})

    def ptr_stale(root):   # Beta Q5: no id, the fixed grader gives stale from the pointer alone
        fixed_prom(root)
        write_json(root / "diagnostics_outputs" / "compare_models_summary_latest.json",
                   {"run_id": "S88", "diagnostics_run_id": RID_B, "models": {"catboost": {}},
                    "winner_best_effort": {"model_type": "catboost", "score": 0.9}})
    def corrupt_legacy(root):   # Beta R2 Q4: invalid UTF-8 in the report (14815a7: PROCEED_WITH_NOTE / absent)
        policies(root)
        preset_registry(root, "neural_net")
        p = root / "diagnostics_outputs" / "training_diagnostics.json"
        p.write_bytes(b'{"model_type": "neural_net", "note": "\xff\xfe bad bytes"}')

    def corrupt_fixed(root):
        corrupt_legacy(root)
        policies(root, grading_rules="fixed")

    def corrupt_fixed_winner(root):
        fixed_prom(root)
        (root / "diagnostics_outputs" / "catboost_diagnostics.json").write_bytes(b'{"model_type": "\xff"}')

    def grade_policy(root):   # Beta R2 Q5: stale_report_action grade, no id: the stale report is graded
        policies(root, stale_report_action="grade")
        preset_registry(root, "neural_net")
        write_json(root / "diagnostics_outputs" / "training_diagnostics.json",
                   critical_report("neural_net", diagnostics_run_id=RID_B))

    def grade_policy_fixed(root):
        grade_policy(root)
        policies(root, grading_rules="fixed", stale_report_action="grade")
    mk("legacy_no_dir", legacy, {})
    mk("legacy_dir", legacy, {"models_dir": "models/reinforcement"})
    mk("legacy_expected_id", legacy, {"expected_run_id": RID, "run_started_at": STARTED})
    mk("fixed_promoted", fixed_prom, {"models_dir": None})
    mk("pointer_stale", ptr_stale, {})
    mk("corrupt_legacy", corrupt_legacy, {})                       # fix pass 2, Beta R2 Q4
    mk("corrupt_fixed", corrupt_fixed, {})
    mk("corrupt_fixed_winner", corrupt_fixed_winner, {})
    mk("grade_policy_legacy", grade_policy, {"expected_run_id": RID, "run_started_at": STARTED})   # Beta R2 Q5
    mk("grade_policy_fixed", grade_policy_fixed, {"expected_run_id": RID, "run_started_at": STARTED})
    out = []
    for tag, setup, kw in cases:
        got = []
        for who, mm in (("new", m), ("base", base)):
            root = d / tag / who
            setup(root)
            res, lines = check(mm, root, n=2, **kw)
            reg = json.loads((root / "diagnostics_outputs" / "model_skip_registry.json").read_text())
            got.append(norm({"res": res, "lines": lines, "reg": reg}))
        out.append((tag, got[0] == got[1]))
    return all(o[1] for o in out), str(out)


def g_absent_step5(m, d):
    # compare: child argv, summary, archive and restore
    base_mpo = base_file("meta_prediction_optimizer_anti_overfit.py", d / "base_src" / "mpo.py")
    a = run_compare(d / "new", m["mpo_path"], m["paths"]["td"])
    b = run_compare(d / "base", base_mpo)

    def tree(r):   # every file the compare run left (archive, restore, reports), run-id folder names normalised
        return {re.sub(r"S88_\d+_\d+", "S88", str(p.relative_to(r))): sha(p) for p in sorted(r.rglob("*"))
                if p.is_file() and p.name not in ("argv.jsonl", "driver.py", "stub.py")
                and "compare_models_summary_S88" not in p.name}
    sa, sb = dict(a["summary"]), dict(b["summary"])
    for s in (sa, sb):
        s.pop("run_id")
        s.pop("timestamp_utc")
    compare_ok = (a["rc"] == b["rc"] == "0" and a["argv"] == b["argv"] and sa == sb
                  and tree(a["root"]) == tree(b["root"]))
    # save_model: sidecar and checkpoint bytes, no binding
    mpo_new, lg = _mpo(m, "abs")
    mpo_base, _ = _mpo(dict(m, mpo_path=base_mpo), "absb")
    mdl, X, y = _xgb()
    sides = []
    for tag, mp in (("n", mpo_new), ("b", mpo_base)):
        with cwd(d / f"save_{tag}") as root:
            mp.MultiModelTrainer(device="cpu", logger=lg).save_model(model=mdl, model_type="xgboost",
                                                                      output_dir="models", **SAVE_KW)
            side = json.loads((root / "models" / "best_model.meta.json").read_text())
            side["training_info"].pop("completed_at")
            sides.append((side, sha(root / "models" / "best_model.json")))
    save_ok = sides[0] == sides[1]
    # NN worker without a request: report keys as 14815a7, no staging dir
    tb = base_file("train_single_trial.py", d / "base_tst" / "train_single_trial.py")
    keys = []
    for tag, tp in (("n", REPO / "train_single_trial.py"), ("b", tb)):
        root = d / f"tst_{tag}"
        p = run_tst(tp, root)
        rep = json.loads((root / "diagnostics_outputs" / "training_diagnostics.json").read_text()) if p.returncode == 0 \
            else {}
        keys.append((p.returncode, sorted(rep), [x.name for x in (root / "diagnostics_outputs").iterdir()
                                                 if x.is_dir()] if (root / "diagnostics_outputs").exists() else None))
    tst_ok = keys[0] == keys[1] and keys[0][0] == 0
    return compare_ok and save_ok and tst_ok, f"compare {compare_ok}; save_model {save_ok}; NN worker {tst_ok} {keys[0][0]}"


# ── fix pass 2 gates (Beta X-W3-BINDING R2 Q2, Q3, Q5) ─────────────────────

def g_nn_fallback_no_publish(m, d):
    """Beta R2 Q2: the NN worker exits 0 with its staged pair complete but its metrics JSON lacks the required
    keys; inline fallback allowed. The real single-shot path completes on the inline model; nothing is published
    (also with a leftover identity.pending.json of the rejected worker, and with a pending record this process did
    not write); a later flag is unbound. Positive control: an accepted worker's pair publishes and binds."""
    import numpy as np
    import subprocess as _sp
    mpo, lg = _mpo(m, "nnfb")
    td = m["td"]
    out = []
    for tag in ("rejected", "rejected_leftover", "accepted", "accepted_foreign_pending"):
        root = (d / tag).resolve()
        with cwd(root), td_as(td):
            req = td.w3_request(RID, STARTED, "nn-1", "neural_net")
            st = Path(req["stage_dir"]).resolve()
            accept = tag.startswith("accepted")

            def worker(cmd, *a, **k):   # the train_single_trial.py worker, as built: staged pair, public copies
                if "train_single_trial.py" not in cmd:
                    return real_run(cmd, *a, **k)
                ckb = b"nn checkpoint bytes " + tag.encode()
                os.makedirs("models/reinforcement", exist_ok=True)
                st.mkdir(parents=True, exist_ok=True)
                (st / "best_model.pth").write_bytes(ckb)
                Path("models/reinforcement/neural_net_trial-1.pth").write_bytes(ckb)
                rep = json.dumps(critical_report("neural_net", diagnostics_run_id=RID, diagnostics_run_started_at=STARTED,
                                                 binding_model_id="nn-1",
                                                 checkpoint_sha256=hashlib.sha256(ckb).hexdigest()), indent=2)
                for p in (st / "neural_net_diagnostics.json", Path("diagnostics_outputs/neural_net_diagnostics.json"),
                          Path("diagnostics_outputs/training_diagnostics.json")):
                    p.write_text(rep)
                if tag == "rejected_leftover":   # a pending record of the rejected worker's files is on disk
                    td.w3_write_pending(req, "neural_net", "best_model.pth", "neural_net_diagnostics.json", True)
                js = ({"r2": 0.5, "normalize_features": True, "use_leaky_relu": True, "scaler_shape": [4]} if accept
                      else {"r2": 0.5})
                return _sp.CompletedProcess(cmd, 0, json.dumps(js) + "\n", "")

            class Tr:   # the inline trainer (the fallback): an in-memory model, no checkpoint, no pair
                def train_model(self, *a, **k):
                    return {"model": object(), "model_type": "neural_net", "hyperparameters": {},
                            "metrics": {"r2": 0.1, "train_mse": 0.1, "val_mse": 0.1}}
            rng = np.random.default_rng(0)
            opt = object.__new__(mpo.AntiOverfitMetaOptimizer)
            opt.__dict__.update(model_type="neural_net", n_trials_total=1, X_train_val=rng.normal(size=(50, 4)),
                                y_train_val=rng.normal(size=50), output_dir="models/reinforcement", logger=lg,
                                _w3_req=req, _cli_enable_diagnostics=True, _cli_diagnostics_level="legacy",
                                _allow_inline_nn_fallback=True, trainer=Tr(), save_best_model=lambda: None)
            real_run = _sp.run
            _sp.run = worker
            try:
                opt._run_single_model()
            finally:
                _sp.run = real_run
            if tag == "accepted_foreign_pending":   # another pending record replaces this process's own
                rec = json.loads((st / "identity.pending.json").read_text())
                td.w3_atomic_write_json(str(st / "identity.pending.json"), dict(rec, recorded_at="2000-01-01T00:00:00Z"))
            pending_after_run = (st / "identity.pending.json").exists()
            ok_pub, why = mpo._w3_publish_final(opt, "neural_net")
        policies(root)
        preset_registry(root, "neural_net")
        _, lines = check(m, root, binding_run_id=RID, run_started_at=STARTED)
        ln = lines[-1] if lines else {}
        out.append(dict(tag=tag, inline=opt.best_checkpoint_path if hasattr(opt, "best_checkpoint_path") else None,
                        record=bool(getattr(opt, "_w3_record", None)), pending=pending_after_run, published=ok_pub,
                        identity=(st / "identity.json").exists(), lines=len(lines), binding=ln.get("binding"),
                        why=why[:60]))
    o = {x["tag"]: x for x in out}
    ok = (all(x["lines"] == 1 for x in out)
          and not o["rejected"]["record"] and not o["rejected"]["pending"] and o["rejected"]["inline"] is None
          and all(not o[t]["published"] and not o[t]["identity"] and o[t]["binding"] == "unbound"
                  for t in ("rejected", "rejected_leftover", "accepted_foreign_pending"))
          and o["rejected_leftover"]["pending"]
          and o["accepted"]["record"] and o["accepted"]["published"] and o["accepted"]["identity"]
          and o["accepted"]["binding"] == "bound")
    return ok, str(out)


def g_fixed_winner_reread(m, d):
    """Beta R2 Q3: fixed grading, critical NN winner of this run; the public winner report is replaced by another
    run's after the graded read (and the same path is read again by anyone). The flag carries the graded bytes'
    sha256 and binds to the graded run's pair; the per-family view of the winner is the graded read."""
    thc = m["thc"]
    root = d / "r"
    policies(root, grading_rules="fixed")
    preset_registry(root, "neural_net")
    stage_pair(m["td"], root, RID, "cmp-neural_net", "neural_net", fixed_report("neural_net", True), b"nnA",
               public=("family",))
    fam = root / "diagnostics_outputs" / "neural_net_diagnostics.json"
    graded_sha = sha(fam)
    write_json(root / "diagnostics_outputs" / "compare_models_summary_latest.json",
               {"run_id": "S88_a", "diagnostics_run_id": RID, "models": {"neural_net": {}},
                "winner_best_effort": {"model_type": "neural_net", "score": 0.9}})
    orig = thc.report_freshness
    state = {"n": 0}

    def hook(report, path, run_ctx=None, *a, **k):
        r = orig(report, path, run_ctx, *a, **k)
        if state["n"] == 0 and run_ctx is not None and str(path).endswith("neural_net_diagnostics.json"):
            state["n"] = 1   # run B rewrites the public family report, and that path is read again
            write_json(Path(path), fixed_report("neural_net", True, diagnostics_run_id=RID_B))
            thc._w3_load_report(path, run_ctx, "fixed")
        return r
    thc.report_freshness = hook
    try:
        res, lines = check(m, root, binding_run_id=RID, run_started_at=STARTED)
    finally:
        thc.report_freshness = orig
    ln = lines[-1] if lines else {}
    st = (root / "diagnostics_outputs" / RID / "cmp-neural_net").resolve()
    pf = res[0].get("per_family", {}).get("neural_net", {})
    ok = (state["n"] == 1 and res[0]["severity"] == "critical" and pf.get("severity") == "critical"
          and len(lines) == 1 and ln.get("binding") == "bound" and ln.get("report_sha256") == graded_sha
          and ln.get("checkpoint_path") == str(st / "best_model.pth"))
    return ok, (f"swapped {state['n']}; per_family {pf.get('severity')}; {ln.get('binding')}: "
                f"{ln.get('binding_reason')}; sha graded {ln.get('report_sha256') == graded_sha}")


def g_pointer_swap(m, d):
    """Beta R2 Q5: the compare pointer is replaced right after it was read. The pointer that passed the id check
    routes: A's healthy NN winner is graded; B's critical CatBoost is not chosen; no flag line."""
    thc = m["thc"]
    root = d / "r"
    policies(root, grading_rules="fixed")
    preset_registry(root, "catboost")
    stage_pair(m["td"], root, RID, "cmp-neural_net", "neural_net", fixed_report("neural_net", False), b"nn",
               public=("family",))
    stage_pair(m["td"], root, RID, "cmp-catboost", "catboost", fixed_report("catboost", True), b"cb", ext=".cbm",
               public=("family",))
    ptr = root / "diagnostics_outputs" / "compare_models_summary_latest.json"
    write_json(ptr, {"run_id": "S88_a", "diagnostics_run_id": RID, "models": {"neural_net": {}},
                     "winner_best_effort": {"model_type": "neural_net", "score": 0.9}})
    orig = thc._load_json
    state = {"n": 0}

    def hook(path):
        r = orig(path)
        if state["n"] == 0 and str(path).endswith("compare_models_summary_latest.json"):
            state["n"] = 1   # run B's pointer replaces A's right after the read
            write_json(Path(path), {"run_id": "S88_b", "diagnostics_run_id": RID_B, "models": {"catboost": {}},
                                    "winner_best_effort": {"model_type": "catboost", "score": 0.9}})
        return r
    thc._load_json = hook
    try:
        res, lines = check(m, root, binding_run_id=RID, run_started_at=STARTED)
    finally:
        thc._load_json = orig
    r = res[0]
    ok = (state["n"] == 1 and r["model_type"] == "neural_net" and r["severity"] == "ok" and not lines
          and r.get("compare_run_id") == "S88_a")
    return ok, f"swapped {state['n']}; model {r['model_type']} severity {r['severity']} lines {len(lines)}"


def g_strict_stale_grade_policy(m, d):
    """Beta R2 Q5: stale_report_action 'grade'. With WATCHER's id a stale report (no id, another id) is never
    graded, in both graders: PROCEED_WITH_NOTE / stale, no count advance, no flag. With no id: as 14815a7 (the
    stale report is graded; W3.absent_thc holds the byte-for-byte side)."""
    out = []
    for rules in ("legacy", "fixed"):
        for tag, rep in (("other_id", critical_report("neural_net", diagnostics_run_id=RID_B)),
                         ("no_id", critical_report("neural_net"))):
            root = d / f"{rules}_{tag}"
            policies(root, grading_rules=rules, stale_report_action="grade")
            preset_registry(root, "neural_net")
            write_json(root / "diagnostics_outputs" / "training_diagnostics.json", rep)
            res, lines = check(m, root, binding_run_id=RID, run_started_at="2000-01-01T00:00:00+00:00")
            reg = json.loads((root / "diagnostics_outputs" / "model_skip_registry.json").read_text())
            out.append((rules, tag, res[0]["action"], res[0]["severity"], len(lines),
                        reg["neural_net"]["consecutive_critical"], bool(res[0].get("stale_report_action_overridden"))))
    # no id, same policy and report: graded (14815a7 behaviour), the count moves
    root = d / "noid"
    policies(root, stale_report_action="grade")
    preset_registry(root, "neural_net")
    write_json(root / "diagnostics_outputs" / "training_diagnostics.json",
               critical_report("neural_net", diagnostics_run_id=RID_B))
    res, lines = check(m, root, expected_run_id=RID)
    noid = (res[0]["severity"], res[0]["action"], len(lines))
    ok = (all(o[2] == "PROCEED_WITH_NOTE" and o[3] == "stale" and o[4] == 0 and o[5] == 1 and o[6] for o in out)
          and noid[0] == "critical" and noid[2] == 1)
    return ok, f"{out}; no id {noid}"


# ── installed WATCHER gate ──────────────────────────────────────────────────

DRIVER5 = r'''
import contextlib, io, json, os, sys, types
sc, root, wpath, mdir, mode = sys.argv[1:6]
sys.path[:0] = [sc]
import importlib.util as u
s = u.spec_from_file_location("wa_w3", wpath)
W = u.module_from_spec(s); sys.modules["wa_w3"] = W; s.loader.exec_module(W)
cfg = W.WatcherConfig(use_llm=False, manifests_dir=mdir,
                      history_file=os.path.join(root, "h.json"), log_file=os.path.join(root, "d.jsonl"),
                      pid_file=os.path.join(root, "p.pid"), daemon_state_file=os.path.join(root, "s.json"),
                      pending_approval_file=os.path.join(root, "pa.json"), halt_file=os.path.join(root, "halt"))
agent = W.WatcherAgent(cfg)
agent.llm_lifecycle = agent.llm_router = None
argvs = []
class Done:
    returncode, stdout, stderr = 0, "", ""
def stream(cmd, step, timeout):
    argvs.append([step, list(cmd)]); return Done()
agent._ensure_execution_set = lambda p=None: object()
agent._run_preflight_check = lambda s: (True, "W3-DRV")
agent._run_step_streaming = stream
agent._run_post_step_cleanup = lambda *a, **k: None
agent._find_results = lambda *a, **k: {"from_disk": True}
fresh = mode == "skip"
W.check_output_freshness = lambda s: (fresh, "W3-DRV", False)
W.p05_freeze_dataset = lambda run_label="watcher": types.SimpleNamespace(
    path="daily3.json", sha256="0" * 64, size=0, record_count=0, manifest_id="W3-DRV")
W.p05_resolve_dataset_path = lambda p: str(p)
import miner.dataset_authority as _ds
for n, v in (("_active_execution_set", lambda: None), ("load_provisioning_nodes", lambda *a, **k: []),
             ("fleet_preflight", lambda *a, **k: []), ("resolve_absent_fleet_status", lambda *a, **k: "NOT_APPLICABLE"),
             ("default_provisioning_manifest_path", lambda *a, **k: "<W3-DRV>"),
             ("write_run_provenance", lambda *a, **k: None)):
    setattr(_ds, n, v)
agent.evaluate_results = lambda step, results, n: (types.SimpleNamespace(recommended_action="proceed", confidence=0.9), {})
def execute(decision, context):
    agent.current_step += 1; return True
agent.execute_decision = execute
agent._invalidate_step_freshness = lambda step: None
agent._build_retry_params = lambda health, params: dict(params or {})
health_calls, resets, results_seen = [], [], []
def chk(*a, **k):
    health_calls.append(dict(k))
    h = {"action": "PROCEED", "model_type": "neural_net", "severity": "ok", "issues": [], "suggested_fixes": [],
         "confidence": 0.9}
    if mode == "retry" and len(health_calls) == 1:
        h.update(action="RETRY", severity="critical")
    if mode == "stale":
        h.update(action="PROCEED_WITH_NOTE", severity="stale")
        if k.get("binding_run_id"):
            h["binding_run_id"] = k["binding_run_id"]
    return h
import time as _t
ft = types.SimpleNamespace(**{k: getattr(_t, k) for k in dir(_t) if not k.startswith("_")}); ft.sleep = lambda s: None
W.__dict__.update(check_safety=lambda: True, notify_telegram=lambda m, *a, **k: None, PROGRESS_DISPLAY_AVAILABLE=False,
                  TRAINING_HEALTH_CHECK_AVAILABLE=True, check_training_health=chk,
                  reset_skip_registry=lambda m: resets.append(m), time=ft)
orig_run_step = agent.run_step
def run_step(step, params=None, **kw):
    r = orig_run_step(step, params, **kw); results_seen.append(dict(r) if isinstance(r, dict) else r); return r
agent.run_step = run_step
params = {"binding_run_id": "caller-chosen-id"} if mode == "caller" else None
os.chdir(root)
with contextlib.redirect_stdout(io.StringIO()):
    ret = agent.run_pipeline(start_step=5, end_step=5, params=params)
print("RESULT " + json.dumps({"argvs": argvs, "health": health_calls, "resets": resets, "results": results_seen},
                             default=str))
'''


def run_drv(d, wpath, mdir, mode, tag):
    d.mkdir(parents=True, exist_ok=True)
    drv = d / "drv5.py"
    if not drv.exists():
        drv.write_text(DRIVER5)
    root = d / "w" / tag
    root.mkdir(parents=True)
    p = subprocess.run([PY, "-B", str(drv), str(REPO), str(root), str(wpath), str(mdir), mode], cwd=str(REPO),
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"),
                       capture_output=True, text=True, timeout=600)
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    if p.returncode != 0 or not line:
        raise RuntimeError(f"{tag}: driver rc {p.returncode}: {p.stderr[-1200:]}")
    return json.loads(line[0][7:])


def _arg(argv, k):
    return argv[argv.index(k) + 1] if k in argv else None


def watcher_checks(d, wpath, mdir):
    """The installed-code behaviour. Returns (ok, detail)."""
    out = {}
    p = run_drv(d, wpath, mdir, "plain", "plain")
    a5 = [a for s, a in p["argvs"] if s == 5]
    rid = _arg(a5[0], "--binding-run-id") if a5 else None
    out["plain"] = (len(a5) == 1 and bool(rid) and re.fullmatch(r"s5-\d{8}T\d{6}Z-[0-9a-f]{8}", rid or "") is not None
                    and _arg(a5[0], "--binding-run-started-at") == p["health"][0].get("run_started_at")
                    and p["health"][0].get("binding_run_id") == rid and p["results"][0].get("step5_run_id") == rid
                    and p["results"][0].get("from_disk") is True)
    r = run_drv(d, wpath, mdir, "retry", "retry")
    ids = [_arg(a, "--binding-run-id") for s, a in r["argvs"] if s == 5]
    out["retry"] = (len(ids) == 2 and ids[0] != ids[1] and [h.get("binding_run_id") for h in r["health"]] == ids)
    c = run_drv(d, wpath, mdir, "caller", "caller")
    cid = [_arg(a, "--binding-run-id") for s, a in c["argvs"] if s == 5]
    out["caller_overwritten"] = len(cid) == 1 and cid[0] not in (None, "caller-chosen-id") \
        and "caller-chosen-id" not in json.dumps(c["argvs"])
    k = run_drv(d, wpath, mdir, "skip", "skip")
    out["skip_no_id"] = not [a for s, a in k["argvs"] if s == 5] and not k["health"] \
        and "step5_run_id" not in json.dumps(k["results"])
    st = run_drv(d, wpath, mdir, "stale", "stale")
    out["stale_no_reset"] = st["resets"] == [] and st["health"][0].get("binding_run_id")
    return all(bool(v) for v in out.values()), out


def owner_installed():
    man = json.loads((REPO / "agent_manifests" / "reinforcement.json").read_text())["default_params"]
    src = (REPO / "agents" / "watcher_agent.py").read_text()
    return "binding_run_id" in man and "[W3 2026-11-27]" in src


def g_installed_watcher(m, d):
    if not owner_installed():
        return "SKIP", ("owner patch docs/drafts/unit_w/W3_OWNER_LAND.patch not installed (manifest has no "
                        "binding_run_id / watcher has no [W3] block)")
    wpath = REPO / "agents" / "watcher_agent.py"
    mdir = REPO / "agent_manifests"
    ok, detail = watcher_checks(d / "installed", wpath, mdir)
    # negative control: the installed watcher with the id dropped from the health call must fail
    neg = d / "neg" / "watcher_agent.py"
    neg.parent.mkdir(parents=True)
    src = wpath.read_text()
    anchor = '**({"binding_run_id": _w3_id,'
    if src.count(anchor) != 1:
        return False, "negative control anchor not found exactly once"
    neg.write_text(src.replace(anchor, '**({"binding_run_id_dropped": _w3_id,'))
    try:
        nok, ndetail = watcher_checks(d / "neg_run", neg, mdir)
    except Exception as e:
        nok, ndetail = False, repr(e)[:200]
    # absent is today: with the 14815a7 manifest (no binding keys) the installed watcher == the 14815a7 watcher
    bm = d / "base_manifests"
    shutil.copytree(mdir, bm)
    base_file("agent_manifests/reinforcement.json", bm / "reinforcement.json")
    bw = base_file("agents/watcher_agent.py", d / "base_w" / "watcher_agent.py")
    x = run_drv(d, wpath, bm, "plain", "abs_new")
    y = run_drv(d, bw, bm, "plain", "abs_base")
    xs = run_drv(d, wpath, bm, "stale", "abs_new_stale")
    ys = run_drv(d, bw, bm, "stale", "abs_base_stale")
    absent = (x["argvs"] == y["argvs"] and x["health"] == y["health"] and x["results"] == y["results"]
              and xs["resets"] == ys["resets"] == ["neural_net"])
    return ok and not nok and absent, f"installed {detail}; negative control red {not nok}; absent-is-today {absent}"


# ── registry and mutants ────────────────────────────────────────────────────

GATES = {
    "W3.bound_nn_legacy": g_bound_nn_legacy, "W3.strict_stale": g_strict_stale,
    "W3.report_sha_link": g_report_sha_link, "W3.staged_ck_changed": g_staged_ck_changed,
    "W3.nn_report_ck": g_nn_report_ck, "W3.override_family": g_override_family,
    "W3.unbound_tree_legacy": g_unbound_tree_legacy, "W3.fixed_promoted": g_fixed_promoted,
    "W3.fixed_leftover_pointer": g_fixed_leftover_pointer, "W3.report_swapped": g_report_swapped,
    "W3.read_write_errors": g_read_write_errors, "W3.deadline": g_deadline, "W3.publish_recheck": g_publish_recheck,
    "W3.nn_writer_pair": g_nn_writer_pair, "W3.tree_writer_pair": g_tree_writer_pair,
    "W3.tree_writer_pair_lightgbm": g_tree_writer_pair_lightgbm,
    "W3.tree_writer_pair_catboost": g_tree_writer_pair_catboost,
    "W3.interleaved_writers": g_interleaved_writers, "W3.write_errors_step5": g_write_errors_step5,
    "W3.compare_child_exit1": g_compare_child_exit1, "W3.compare_publish_error": g_compare_publish_error,
    "W3.full_no_request": g_full_no_request, "W3.one_report_per_family": g_one_report_per_family,
    "W3.runner_bound_then_unbound": g_runner_bound_then_unbound, "W3.runner_ll8_handoff": g_runner_ll8_handoff,
    "W3.absent_thc": g_absent_thc, "W3.absent_step5": g_absent_step5, "W3.installed_watcher": g_installed_watcher,
    # fix pass 2 (Beta X-W3-BINDING R2)
    "W3.nn_fallback_no_publish": g_nn_fallback_no_publish, "W3.fixed_winner_reread": g_fixed_winner_reread,
    "W3.pointer_swap": g_pointer_swap, "W3.strict_stale_grade_policy": g_strict_stale_grade_policy,
}

# (id, file key, old, new, gate). `old` must occur exactly once in the file.
MUTANTS = [
    ("strict_off", "thc", "        if run_ctx.get('strict_run_id'):   # [W3, owner decision 2",
     "        if False:   # [W3, owner decision 2", "W3.strict_stale"),
    ("no_report_sha_link", "thc", "                and rec.get('report_sha256') == gr.get('report_sha256')):",
     "                ):", "W3.report_sha_link"),
    ("no_ck_hash_link", "thc", "    if got != rec.get('checkpoint_sha256'):", "    if False:", "W3.staged_ck_changed"),
    ("no_nn_report_ck_link", "thc",
     "    if family == 'neural_net' and gr.get('checkpoint_sha256') != rec.get('checkpoint_sha256'):",
     "    if False:", "W3.nn_report_ck"),
    ("winner_tuple_always", "thc",
     "        run_ctx['graded_report'] = (_w3_family_tuples.get(promoted) if promoted is not None\n"
     "                                    else _w3_winner_tuple)",
     "        run_ctx['graded_report'] = _w3_winner_tuple", "W3.fixed_promoted"),
    ("pointer_kept", "thc", "            ptr = None\n    if _w3_bind and ptr", "            pass\n    if _w3_bind and ptr",
     "W3.fixed_leftover_pointer"),
    ("no_deadline", "thc", "        t.join(deadline)", "        t.join()", "W3.deadline"),
    ("binding_after_nothing", "thc",
     "    _w3_fields = _bind_flagged_model(model_type, run_ctx) if _w3 else None   # [W3] before the append",
     "    _w3_fields = None", "W3.bound_nn_legacy"),
    ("rehash_report_file", "thc",
     "                and rec.get('report_sha256') == gr.get('report_sha256')):",
     "                and rec.get('report_sha256') == _sha256_file(gr.get('report_path'))):", "W3.report_swapped"),
    ("publish_any_rc", "mpo", "                if rc == 0:\n                    _w3_ok, _w3_why = _w3_td.w3_publish(",
     "                if True:\n                    _w3_ok, _w3_why = _w3_td.w3_publish(", "W3.compare_child_exit1"),
    ("stage_from_public", "mpo",
     "                (model.booster_ if model_type == 'lightgbm' else model).save_model(_w3_staged)\n"
     "                _w3_sh.copyfile(_w3_staged, str(checkpoint_path))",
     "                (model.booster_ if model_type == 'lightgbm' else model).save_model(str(checkpoint_path))\n"
     "                _w3_sh.copyfile(str(checkpoint_path), _w3_staged)", "W3.interleaved_writers"),
    # [W3 fix pass 1] one family dropped from the staged-save branch: it falls to the plain public save, no staged
    # checkpoint, no record, no sidecar block
    ("staged_save_no_lightgbm", "mpo", "        elif w3_req and model_type in ('xgboost', 'lightgbm', 'catboost'):",
     "        elif w3_req and model_type in ('xgboost', 'catboost'):", "W3.tree_writer_pair_lightgbm"),
    ("staged_save_no_catboost", "mpo", "        elif w3_req and model_type in ('xgboost', 'lightgbm', 'catboost'):",
     "        elif w3_req and model_type in ('xgboost', 'lightgbm'):", "W3.tree_writer_pair_catboost"),
    ("full_id_turns_w3_on", "mpo",
     "            'binding_run_id': _td.w3_safe_id(getattr(args, 'binding_run_id', None)),  # [W3] None = off",
     "            'binding_run_id': _td.w3_safe_id(getattr(args, 'binding_run_id', None) or "
     "(_ch14_opts or {}).get('run_id')),", "W3.full_no_request"),
    ("nn_pair_unchecked", "td", '    if require_report_checkpoint and rep.get("checkpoint_sha256") != ck_sha:',
     "    if False:", "W3.nn_writer_pair"),
    ("publish_no_recheck", "td",
     '        if w3_sha256_file(os.path.join(stage_dir, rec["checkpoint_file"])) != rec["checkpoint_sha256"]:',
     "        if False:", "W3.publish_recheck"),
    ("runner_old_rule", "wrr", '    if latest is not None and latest.get("binding_schema"):', "    if False:",
     "W3.runner_bound_then_unbound"),
    ("runner_no_explicit_path", "wrr", '("flag_checkpoint_path", flag_checkpoint_path),',
     '("flag_checkpoint_path", None),', "W3.runner_ll8_handoff"),
]

# [W3 fix pass 2] Beta X-W3-BINDING R2 Q2 to Q5
MUTANTS += [
    ("nn_pending_before_validate", "mpo", "            # [S96A] Validate enriched checkpoint via subprocess JSON output",
     "            if getattr(self, '_w3_req', None):\n"
     "                import training_diagnostics as _w3_td\n"
     "                self._w3_record = _w3_td.w3_write_pending(self._w3_req, 'neural_net', 'best_model.pth',\n"
     "                                                          'neural_net_diagnostics.json', True)\n"
     "            # [S96A] Validate enriched checkpoint via subprocess JSON output", "W3.nn_fallback_no_publish"),
    ("publish_any_pending", "mpo",
     "    if not _w3_acc:\n        return False, \"no pair recorded by the accepted final path\"\n",
     "    if False:\n        return False, \"no pair recorded by the accepted final path\"\n",
     "W3.nn_fallback_no_publish"),
    ("publish_expect_ignored", "td", "            if _diff:\n", "            if False:\n", "W3.nn_fallback_no_publish"),
    ("winner_reread", "thc", "            if _w3_bind and fp == report_path:", "            if False:",
     "W3.fixed_winner_reread"),
    ("winner_tuple_from_dict", "thc", "                                    else _w3_winner_tuple)",
     "                                    else run_ctx.get('_w3_loaded', {}).get(report_path))",
     "W3.fixed_winner_reread"),
    ("pointer_reread", "thc", "        cs = _w3_cs if _w3_cs is not None else _load_json(ptr)",
     "        cs = _load_json(ptr)", "W3.pointer_swap"),
    ("strict_grade_policy_off", "thc", "    if action == 'grade' and strict:", "    if False:",
     "W3.strict_stale_grade_policy"),
    ("strict_grade_policy_always", "thc", "    if action == 'grade' and strict:", "    if action == 'grade':",
     "W3.absent_thc"),
    ("decode_error_legacy_noid", "thc",
     "if run_ctx.get('binding_run_id')\n            else (json.JSONDecodeError, IOError))",
     "if run_ctx.get('binding_run_id')\n            else (json.JSONDecodeError, IOError, UnicodeDecodeError))",
     "W3.absent_thc"),
    ("decode_error_fixed_noid", "thc", "if _w3_bind\n            else (json.JSONDecodeError, IOError))",
     "if _w3_bind\n            else (json.JSONDecodeError, IOError, UnicodeDecodeError))", "W3.absent_thc"),
]

FILES = {"thc": "training_health_check.py", "td": "training_diagnostics.py",
         "wrr": "scripts/diagnostics/why_report_runner.py", "mpo": "meta_prediction_optimizer_anti_overfit.py"}


def run_gate(name, m, d):
    d = Path(d).resolve()
    d.mkdir(parents=True, exist_ok=True)
    try:
        return GATES[name](m, d)
    except Exception as e:
        import traceback
        return False, f"gate raised {e!r}: {traceback.format_exc()[-900:]}"


@pytest.mark.parametrize("name", list(GATES))
def test_gate(name, tmp_path):
    ok, detail = run_gate(name, mods(), tmp_path)
    if ok == "SKIP":
        pytest.skip(detail)
    assert ok is True, f"{name}: {detail}"


@pytest.mark.parametrize("mid,key,old,new,gate", MUTANTS, ids=[x[0] for x in MUTANTS])
def test_mutant(mid, key, old, new, gate, tmp_path):
    src = (REPO / FILES[key]).read_text()
    assert src.count(old) == 1, f"{mid}: anchor found {src.count(old)} times in {FILES[key]}"
    # the mutant copy keeps the tree layout so module-relative paths (wrr REPO) still resolve under data/
    mdir = tmp_path / "code"
    dest = mdir / FILES[key]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(src.replace(old, new))
    ok, detail = run_gate(gate, mods({key: dest}), tmp_path / "run")
    assert ok is False, f"mutant {mid} NOT killed by {gate}: {detail}"
    assert not str(detail).startswith("gate raised"), f"mutant {mid}: gate crashed, not a clean red: {detail}"
