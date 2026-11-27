#!/usr/bin/env python3
"""
tests/test_why_report_runner.py - self-test for unit W1 (scripts/diagnostics/why_report_runner.py) and its
additive change to why_report_flags.py (attempts file, status, FAILED-first summary, attach sha256).
Plan: PLAN_LAND_LL7_LL8_20261126 §6.6, §6.7, §6.9; lead task 2026-11-26 (decisions D-1..D-5).

Fixture: one planted-signal LightGBM trained by the real Step 5 CLI on CPU
(tests/test_ll8_attribution_cycle.planted_fixture, imported, not copied), and a synthetic walk-forward root
laid out as scripts/diagnostics/e2e_walk_forward.py writes it (origin_<i>/origin_summary.json with
frozen_families, votes, grades, labels, block; step3/survivors_with_scores.json; frozen/s42/lightgbm;
vote/<arm>/pools.jsonl; grade dirs with per_draw.jsonl + SHA256SUMS; run_record.json frame). The votes
matrix comes from tests/test_ll8_attribution_cycle.write_votes through an injected votes builder: the
synthetic vote dirs carry no vote-harness record, so the default builder (e2e_vote_npz.py child) is run on
the fixture only for its failure path (spec_build_error); its success on a real walk-forward block is a
separate, recorded probe (READY note), not a gate here.

Gates: W1.flags_baseline_identity (no attempts file => every why_report_flags function returns what
a85adb1's does), W1.flags_status, W1.success_attached (latest matching block, report attached, flag
closed, attached sha256 == file sha256, report checkpoint == the model's), W1.parity_hand_run,
W1.attach_sha_bound (fake LL-8), one gate per failure reason W1.fail.<reason>, W1.one_attempt_per_call,
W1.failed_listed_first, W1.no_exception_escapes, W1.cli, W1.except_handlers_surface,
W1.no_write_outside (with a negative control), W1.settings_unchanged, W1.votes_from_unlearned (second pass:
V rebuilt from the unlearned arm of the same pool size, grade dirs still the family arm's, the source recorded
in spec, result and attach record; fallback to the family arm when no unlearned arm exists).
Mutants (source edits of a copy of the runner, each applied exactly once, a marker proves the mutated line
ran, credited only when its named gate is green unmutated and red mutated): close_on_failure,
swallow_exception, skip_attach_sha, write_policy, votes_family_arm, retry_loop.

Third pass (Beta X-W1-RUNNER R1): gates W1.log_append_failure (function, single-flag CLI main, run_open_flags),
W1.child_group_cleanup (killed / normal exit / interrupted wait), W1.fail.no_graded_draws, W1.pooled_only,
W1.analysis_keys, W1.grade_binding, W1.out_root_boundary, W1.cli_no_bytecode (plain CLI in a clean scratch
copy, with a negative control), W1.attempts_unreadable, W1.g_q4_scaler (relocated NN through the runner);
mutants log_failure_unprotected, no_group_kill, no_graded_draws_unchecked, analysis_any_key, grade_binding_off.

CPU only. Writes only under --out (must lie under <this tree>/data/).
  CUDA_VISIBLE_DEVICES="" taskset -c 0,1,7 nice -n 10 python3 -u -B tests/test_why_report_runner.py --out data/w1_build/t_<stamp>
"""
import argparse
import ast
import hashlib
import importlib.util
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

RUNNER_REL = "scripts/diagnostics/why_report_runner.py"
BASE_REV = "a85adb1"
FAM = "lightgbm"
FID = "lightgbm@2026-11-26T01:00:00+00:00"
FID_OLD = "neural_net@2026-11-25T01:00:00+00:00"
FAST = {"bootstrap": {"B": 400}}           # analysis override for speed (LL-8's own test uses B=400)
RESULTS = []
MARK = '__import__("pathlib").Path(__import__("os").environ["W1_MUTANT_MARK"]).touch()'

# (id, [(old, new)], target gates). Each `old` must occur exactly once in the runner.
MUTANTS = [
    ("close_on_failure",
     [('            res["attempt"] = wrf.record_attempt(flag_id, "failed", root, reason=reason, error_class=ecls,',
       '            log_path and wrf.attach(flag_id, str(log_path), "mutant", root)\n'
       '            res["attempt"] = wrf.record_attempt(flag_id, "failed", root, reason=reason, error_class=ecls,')],
     ["W1.fail.no_graded_block"]),
    ("swallow_exception",
     [('            res["attempt"] = wrf.record_attempt(flag_id, "failed", root, reason=reason, error_class=ecls,',
       '            res["attempt"] = None and wrf.record_attempt(flag_id, "failed", root, reason=reason, error_class=ecls,')],
     ["W1.fail.no_graded_block"]),
    ("skip_attach_sha",
     [('            att = wrf.attach(flag_id, rp, by, root, note="automatic why-report (W1, LL-8); advisory",\n'
       '                             report_sha256=rsha,',
       '            att = wrf.attach(flag_id, rp, by, root, note="automatic why-report (W1, LL-8); advisory",\n'
       '                             report_sha256=None,')],
     ["W1.attach_sha_bound"]),
    ("write_policy",
     [('        family = g["model_type"]',
       '        family = g["model_type"]; open(Path(root or ".") / "watcher_policies.json", "a").write(" ")')],
     ["W1.settings_unchanged"]),
    ("votes_family_arm",
     [('    if un:\n        return dict(un[0], votes_source="unlearned")',
       '    if un and False:\n        return dict(un[0], votes_source="unlearned")')],
     ["W1.votes_from_unlearned"]),
    # third pass (Beta X-W1-RUNNER R1), one per lead item 1-5
    ("log_failure_unprotected",
     [('                res["log_error"] = repr(le)', '                raise')],
     ["W1.log_append_failure"]),
    ("no_group_kill",
     [('            _kill_group(p.pid)                             # every ending: the group\'s leftovers die',
       '            pass')],
     ["W1.child_group_cleanup"]),
    ("no_graded_draws_unchecked",
     [('        if n_cells == 0:', '        if n_cells == 0 and False:')],
     ["W1.fail.no_graded_draws"]),
    ("analysis_any_key",
     [('    bad = sorted(k for k in analysis if k not in ANALYSIS_KEYS)', '    bad = []')],
     ["W1.analysis_keys"]),
    ("grade_binding_off",
     [('        if off:\n            raise RunnerFailure("input_sha_mismatch", f"{gd}: grade not bound',
       '        if off and False:\n            raise RunnerFailure("input_sha_mismatch", f"{gd}: grade not bound')],
     ["W1.grade_binding"]),
    ("retry_loop",
     [('            results.append(run_for_flag(fid, wf_roots, root=root, **kw))',
       '            results.append(run_for_flag(fid, wf_roots, root=root, **kw))\n'
       '            results[-1].get("status") == "failed" and run_for_flag(fid, wf_roots, root=root, **kw)')],
     ["W1.one_attempt_per_call"]),
]


def gate(name, ok, detail=""):
    RESULTS.append({"gate": name, "ok": bool(ok), "detail": str(detail)[:600]})
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {str(detail)[:400]}", flush=True)
    return ok


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_as(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def tree_state(d, skip=()):
    """{relative path: (size, sha256)} of every file under d, pruning the `skip` subtrees."""
    d = Path(d)
    out = {}
    skip = [str(Path(s).resolve()) for s in skip]
    for root, dirs, files in os.walk(d):
        dirs[:] = [x for x in dirs if str((Path(root) / x).resolve()) not in skip]
        for n in files:
            p = Path(root) / n
            if str(p.resolve()) in skip:
                continue
            out[str(p.relative_to(d))] = (p.stat().st_size, sha(p))
    return out


def settings_shas(roots):
    files = [REPO / "watcher_policies.json"] + sorted((REPO / "agent_manifests").glob("*.json"))
    for r in roots:
        files += [Path(r) / "watcher_policies.json"] + sorted((Path(r) / "agent_manifests").glob("*.json"))
    return {str(f): sha(f) for f in files if f.is_file()}


# ---------------------------------------------------------------------------- fixtures --
def build_model(out):
    import test_ll8_attribution_cycle as ll8
    surv, mdir = ll8.planted_fixture(out)
    return Path(surv), Path(mdir)


def train_nn(fx, surv):
    """One neural_net on the planted survivors, by the real Step 5 CLI on CPU, as
    tests/test_ll6_inference_bundle.build_fixtures trains its NN (--normalize-features --use-leaky-relu)."""
    R = Path(fx) / "models_nn"
    R.mkdir()
    shutil.copy(REPO / "train_single_trial.py", R / "train_single_trial.py")   # TODO-E2E-24, as LL-6
    env = dict(os.environ, PYTHONPATH=str(REPO), CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([sys.executable, "-u", "-B", str(REPO / "meta_prediction_optimizer_anti_overfit.py"),
                        "--survivors", str(surv), "--lottery-data", str(Path(fx) / "fx_planted" / "train.json"),
                        "--trials", "1", "--k-folds", "2", "--model-type", "neural_net", "--device", "cpu",
                        "--storage", f"sqlite:///{R}/optuna.db", "--study-name", "w1_nn",
                        "--output-dir", str(R / "models/reinforcement"), "--normalize-features", "--use-leaky-relu"],
                       cwd=R, env=env, capture_output=True, text=True)
    (R / "run.log").write_text(p.stdout + p.stderr)
    if p.returncode != 0:
        raise RuntimeError(f"NN Step 5 failed rc={p.returncode}; see {R}/run.log")
    return R / "models/reinforcement"


def write_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1))


def flag_line(fid, fam, at):
    return {"schema": "ch14_why_report_flag/1", "flag_id": fid, "flagged_at": at, "model_type": fam,
            "consecutive_critical": 3, "threshold": 3, "critical_since": at, "critical_since_known": True,
            "benching": "off"}


DATASET_SHA = "d" * 64                          # the fixture run_record's inputs.dataset.sha256
DATASET_SESSIONS = ["evening", "midday"]


def grade_summary(gdir, arm, pools_sha, b0, n, sessions):
    """The grader summary.json fields W1 binds (e2e_grader.py write_outputs + summary inputs/declared/
    grade_block); returns its sha256 (the origin summary's grades[].summary_sha256 pin)."""
    write_json(gdir / "summary.json", {
        "outputs": {"per_draw.jsonl": {"sha256": sha(gdir / "per_draw.jsonl")}},
        "inputs": {"pools": {"sha256": pools_sha}, "dataset": {"sha256": DATASET_SHA}},
        "declared": {"arms": [{"arm_id": arm, "max_k": 300}]},
        "grade_block": {"start_index": b0, "end_index": b0 + n, "sessions": sessions}})
    return sha(gdir / "summary.json")


def make_origin(wf, i, b0, n, surv, mdir, ck_sha_override=None, unlearned=False, fam=FAM, pooled_only=False):
    """One origin dir as e2e_walk_forward.run_origin writes it (only the fields W1 reads)."""
    import numpy as np
    import test_ll8_attribution_cycle as ll8
    o = wf / f"origin_{i}"
    (o / "step3").mkdir(parents=True)
    shutil.copy(surv, o / "step3" / "survivors_with_scores.json")
    fz = o / "frozen" / "s42" / fam
    fz.mkdir(parents=True)
    meta = json.load(open(mdir / "best_model.meta.json"))
    ck = Path(meta["checkpoint_path"])
    if not ck.exists():
        ck = mdir / ck.name
    shutil.copy(mdir / "best_model.meta.json", fz / "best_model.meta.json")
    shutil.copy(ck, fz / ck.name)
    if ck_sha_override:                               # a different checkpoint: change its bytes
        with open(fz / ck.name, "a") as f:
            f.write("\n")
    shas = {n_: sha(fz / n_) for n_ in ("best_model.meta.json", ck.name)}
    arm = f"F{fam}s42_o{i}_p300"
    vd = o / "vote" / arm
    vd.mkdir(parents=True)
    (vd / "pools.jsonl").write_text(json.dumps({"target": {"combined_index": b0}, "pool": [1, 2, 3]}) + "\n")
    rows = json.load(open(surv))
    src = o / "grade_src"
    src.mkdir()
    votes, gd, _ = ll8.write_votes(src, "o", rows, lambda V, rr, k: np.array(
        [V[rr.integers(20), j] if rr.random() < 0.5 else rr.integers(0, 1000) for j in range(k)]),
        n_draws=n, t_start=b0)
    gsess = "pooled" if pooled_only else "midday"
    gdir = o / "grade" / f"{arm}_{gsess}"
    gdir.parent.mkdir()
    shutil.copytree(gd, gdir)
    gsum = grade_summary(gdir, arm, sha(vd / "pools.jsonl"), b0, n,
                         DATASET_SESSIONS if pooled_only else ["midday"])
    summ = {"label": "DEVELOPMENT", "origin": i,
            "block": {"block": i, "start_index": b0, "end_index": b0 + n, "sessions": {"midday": n // 2,
                                                                                    "evening": n - n // 2}},
            "labels": {"survivors_with_scores_sha256": sha(surv), "rows": len(rows)},
            "frozen_families": {"s42": {fam: {"dir": str(fz), "sha256": shas, "checkpoint_sha256": shas[ck.name],
                                              "sidecar_sha256": shas["best_model.meta.json"]}}},
            "votes": [{"arm_id": arm, "role": "family", "split_seed": 42, "family": fam, "pool_size": 300,
                       "dir": str(vd), "pools_sha256": sha(vd / "pools.jsonl"), "model_type": fam}],
            "grades": [{"arm_id": arm, "role": "family", "split_seed": 42, "family": fam, "pool_size": 300,
                        "session": gsess, "dir": str(gdir), "summary_sha256": gsum}],
            "finished": "2026-11-26T00:00:00+00:00"}
    if unlearned:   # unlearned arms as e2e_walk_forward writes them (role, family None); p20 must not be picked
        for k in (20, 300):
            ud = o / "vote" / f"A0_o{i}_p{k}"
            ud.mkdir(parents=True)
            (ud / "pools.jsonl").write_text(json.dumps({"target": {"combined_index": b0}, "pool": [k]}) + "\n")
            summ["votes"].insert(0, {"arm_id": ud.name, "role": "unlearned", "split_seed": None, "family": None,
                                     "pool_size": k, "dir": str(ud), "pools_sha256": sha(ud / "pools.jsonl"),
                                     "model_type": "unlearned:field_sum"})
    write_json(o / "origin_summary.json", summ)
    return votes


def make_world(base, surv, mdir, flags=((FID, FAM, "2026-11-26T01:00:00+00:00"),), unlearned=False, fam=FAM,
               pooled_only=False):
    """A flags root (diagnostics_outputs, models/reinforcement, policy + manifest copies) and a walk-forward
    root with three origins: 0 and 1 match the model's checkpoint (1 is later), 2 is latest but its
    checkpoint differs (D-1 must pick origin 1)."""
    root, wf = base / "root", base / "wf"
    (root / "diagnostics_outputs").mkdir(parents=True)
    (root / "diagnostics_outputs" / "why_report_flags.jsonl").write_text(
        "".join(json.dumps(flag_line(*f)) + "\n" for f in flags))
    (root / "agent_manifests").mkdir()
    shutil.copy(REPO / "agent_manifests" / "prediction.json", root / "agent_manifests" / "prediction.json")
    shutil.copy(REPO / "watcher_policies.json", root / "watcher_policies.json")
    shutil.copytree(mdir, root / "models" / "reinforcement")
    write_json(wf / "run_record.json", {"kind": "e2e_walk_forward_run_record",
                                        "inputs": {"dataset": {"sha256": DATASET_SHA}},
                                        "frame": {"pool_sizes": [20, 100, 300],
                                                  "grade_sessions": ["pooled"] if pooled_only
                                                  else ["midday", "evening", "pooled"],
                                                  "dataset_sessions": DATASET_SESSIONS}})
    make_origin(wf, 0, 20000, 330, surv, mdir, fam=fam, pooled_only=pooled_only)
    votes1 = make_origin(wf, 1, 20330, 330, surv, mdir, unlearned=unlearned, fam=fam, pooled_only=pooled_only)
    make_origin(wf, 2, 20660, 330, surv, mdir, ck_sha_override=True, fam=fam, pooled_only=pooled_only)
    return root, wf, votes1


def copy_builder(src_npz, mutate=None):
    def b(cand, out, log, deadline):
        import numpy as np
        p = out / "votes.npz"
        with np.load(src_npz) as z:
            d = {k: z[k] for k in z.files}
        if mutate:
            d = mutate(d)
        np.savez(p, **d)
        return p
    return b


# ---------------------------------------------------------------------------- checks --
def attempts_of(wrf, root, fid):
    g = wrf.read_flags(str(root))["groups"][fid]
    return g


def failed_ok(wrf, root, fid, res, reason, before):
    g = attempts_of(wrf, root, fid)
    t = g["attempts"][-1] if g.get("attempts") else {}
    need = ("flag_id", "attempt", "reason", "error_class", "message", "log_path", "started_at", "ended_at")
    ok = (res.get("status") == "failed" and res.get("reason") == reason and g["open"] and not g["attachments"]
          and g.get("status") == "why_report_failed" and g.get("attempt_count") == before + 1
          and t.get("reason") == reason and t.get("outcome") == "failed" and all(k in t for k in need)
          and t.get("attempt") == before + 1 and len(t.get("message") or "") <= 500)
    return ok, (f"res {res.get('status')}/{res.get('reason')} ({str(res.get('message'))[:160]}); flag open "
                f"{g['open']} status {g.get('status')} attempts {g.get('attempt_count')} (before {before})")


FAKE_LL8 = r'''
import json, sys, pathlib
a = sys.argv[1:]
spec = json.load(open(a[a.index("--spec") + 1])); out = pathlib.Path(a[a.index("--out-dir") + 1]); out.mkdir()
rep = out / "attribution_cycle.json"; rep.write_text('{"fake": true}')
rr = out / "run_record.json"; rr.write_text("{}")
rep.write_text(json.dumps({"fake": True, "contrasts": {"blockX/midday": {"draws": DRAWS}}}))
json.dump({"status": "ok", "exit_code": 0, "report_path": str(rep), "run_record_path": str(rr),
           "priorities_path": None, "checkpoint_sha256": spec["expect_checkpoint_sha256"]},
          open(a[a.index("--result-file") + 1], "w"))
'''.replace("DRAWS", "1")
FAKE_LL8_NO_DRAWS = FAKE_LL8.replace('"draws": 1', '"draws": 0')


# ---------------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--runner-file", default=None, help="mutant runs only")
    ap.add_argument("--models-from", default=None, help="reuse a fixture dir (fx/) from an earlier run")
    ap.add_argument("--only", default=None)
    ap.add_argument("--no-mutants", action="store_true")
    ap.add_argument("--boundary", default=None, help="data/ subtree excluded from the tree-unchanged gate "
                                                     "(default --out; it must contain --out)")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if not str(out).startswith(str(REPO / "data") + "/"):
        raise SystemExit("--out must lie under <this tree>/data/")
    boundary = Path(a.boundary).resolve() if a.boundary else out
    if out != boundary and not str(out).startswith(str(boundary) + "/"):
        raise SystemExit("--out must lie inside --boundary")
    out.mkdir(parents=True)
    import test_ll6_inference_bundle as ll6
    ll6.redirect_env_roots(out)
    t0 = time.time()
    only = set(a.only.split(",")) if a.only else None
    want = (lambda g: only is None or g in only)
    repo_before = ll6.boundary_snapshot(boundary)
    import why_report_flags as wrf
    rf_path = Path(a.runner_file).resolve() if a.runner_file else REPO / RUNNER_REL
    W = load_as("why_report_runner_under_test", rf_path)
    if a.runner_file:                     # a mutant copy lives elsewhere: bind it to this tree
        W.REPO, W.LL8, W.VOTE_NPZ = REPO, REPO / "scripts/diagnostics/attribution_cycle.py", \
            REPO / "scripts/diagnostics/e2e_vote_npz.py"
    W.DEFAULT_OUT_BASE = out / "why_reports_default"
    logs = io.StringIO()
    h = logging.StreamHandler(logs)
    h.setLevel(logging.ERROR)
    h.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    logging.getLogger("why_report_runner").addHandler(h)
    timing = {}

    # ---- W1.flags_baseline_identity (no fixture needed) ---------------------------------
    if want("W1.flags_baseline_identity"):
        old_src = subprocess.run(["git", "-C", str(REPO), "show", f"{BASE_REV}:why_report_flags.py"],
                                 capture_output=True, text=True, check=True).stdout
        (out / "base").mkdir()
        (out / "base" / "why_report_flags_a85adb1.py").write_text(old_src)
        old = load_as("wrf_a85adb1", out / "base" / "why_report_flags_a85adb1.py")
        fr = out / "base" / "root"
        (fr / "diagnostics_outputs").mkdir(parents=True)
        (fr / "rep.md").write_text("r")
        lines = [flag_line(FID_OLD, "neural_net", "2026-11-25T01:00:00+00:00"),
                 flag_line(FID, FAM, "2026-11-26T01:00:00+00:00"),
                 {"model_type": "xgboost", "critical_since": None, "flagged_at": "2026-11-24T00:00:00+00:00"},
                 flag_line(FID, FAM, "2026-11-26T02:00:00+00:00")]
        (fr / wrf.FLAGS_PATH).write_text("".join(json.dumps(x) + "\n" for x in lines) + "{bad\n")
        (fr / wrf.ATTACH_PATH).write_text(json.dumps({"schema": wrf.ATTACH_SCHEMA, "flag_id": "xgboost@2026-11-24T00:00:00+00:00",
                                                      "report_path": "rep.md", "by": "m"}) + "\n"
                                          + json.dumps({"flag_id": "orphan@x", "report_path": "z"}) + "\n")
        diffs = []

        def cmp(what, f):
            x, y = f(old), f(wrf)
            if x != y:
                diffs.append(what)
        cmp("read_flags", lambda m: m.read_flags(str(fr)))
        cmp("open_flags", lambda m: m.open_flags(str(fr)))
        cmp("summary_lines", lambda m: m.summary_lines(str(fr)))
        cmp("flag_id_of", lambda m: [m.flag_id_of(x) for x in lines])
        for argv in (["list"], ["list", "--all"], ["list", "--json"], ["list", "--all", "--json"]):
            def cli(m, argv=argv):
                buf, err = io.StringIO(), io.StringIO()
                so, se = sys.stdout, sys.stderr
                sys.stdout, sys.stderr = buf, err
                try:
                    rc = m.main(["--root", str(fr)] + argv)
                finally:
                    sys.stdout, sys.stderr = so, se
                return rc, buf.getvalue(), err.getvalue()
            cmp(f"main {argv}", cli)
        # attach: same record apart from attached_at, on two copies of the root
        recs = []
        for m, tag in ((old, "o"), (wrf, "n")):
            r2 = out / "base" / f"root_{tag}"
            shutil.copytree(fr, r2)
            rec = m.attach(FID, "rep.md", "michael", str(r2), note="n")
            rec.pop("attached_at")
            recs.append(rec)
            errs = []
            for args in ((FID, "rep.md", "michael"), ("nope@x", "rep.md", "m"), (FID_OLD, "missing.md", "m"),
                         (FID_OLD, "rep.md", " ")):
                try:
                    m.attach(*args, root=str(r2))
                    errs.append(None)
                except ValueError as e:
                    errs.append(str(e))
            recs.append(errs)
        if recs[0] != recs[2] or recs[1] != recs[3]:
            diffs.append("attach")
        gate("W1.flags_baseline_identity", not diffs and not (fr / wrf.ATTEMPTS_PATH).exists(),
             f"a85adb1 vs this tree on a root with flags, a legacy line, a bad line, an attachment, an orphan; "
             f"no attempts file; differing: {diffs or 'none'}")

    # ---- W1.flags_status ------------------------------------------------------------------
    if want("W1.flags_status"):
        sr = out / "status_root"
        (sr / "diagnostics_outputs").mkdir(parents=True)
        (sr / "rep.md").write_text("r")
        (sr / wrf.FLAGS_PATH).write_text(json.dumps(flag_line(FID, FAM, "2026-11-26T01:00:00+00:00")) + "\n")
        refused = []
        for kw in ({"flag_id": "nope@x", "outcome": "failed", "reason": "timeout"},
                   {"flag_id": FID, "outcome": "failed"}, {"flag_id": FID, "outcome": "maybe"}):
            try:
                wrf.record_attempt(root=str(sr), **kw)
            except ValueError:
                refused.append(True)
        s0 = wrf.read_flags(str(sr))["groups"][FID]
        wrf.record_attempt(FID, "failed", str(sr), reason="timeout", error_class="X", message="m" * 900)
        s1 = wrf.read_flags(str(sr))["groups"][FID]
        wrf.record_attempt(FID, "ok", str(sr))
        s2 = wrf.read_flags(str(sr))["groups"][FID]
        wrf.attach(FID, "rep.md", "m", str(sr), report_sha256=sha(sr / "rep.md"))
        s3 = wrf.read_flags(str(sr))["groups"][FID]
        try:
            wrf.attach(FID_OLD, "rep.md", "m", str(sr), report_sha256="0" * 64)
            bad_sha = False
        except ValueError:
            bad_sha = True
        gate("W1.flags_status",
             len(refused) == 3 and "status" not in s0 and s1["status"] == "why_report_failed" and s1["open"]
             and s1["attempt_count"] == 1 and len(s1["attempts"][0]["message"]) == 500
             and s2["status"] == "open" and s2["attempt_count"] == 2 and s2["attempts"][1]["attempt"] == 2
             and s3["status"] == "closed" and not s3["open"] and bad_sha
             and s3["attachments"][0]["report_sha256"] == sha(sr / "rep.md"),
             f"refusals {len(refused)}/3; statuses {s0.get('status')} -> {s1['status']} -> {s2['status']} -> "
             f"{s3['status']}; message cut to {len(s1['attempts'][0]['message'])}; wrong sha refused {bad_sha}")

    # ---- fixture --------------------------------------------------------------------------
    need_fx = only is None or any(g.startswith(("W1.success", "W1.parity", "W1.fail", "W1.one", "W1.failed",
                                                "W1.no_", "W1.cli", "W1.attach", "W1.settings", "W1.votes",
                                                "W1.flag_", "W1.log", "W1.child", "W1.pooled", "W1.analysis",
                                                "W1.grade", "W1.out_root", "W1.attempts", "W1.g_q4")) for g in only)
    if not need_fx:
        return finish(out, t0, [], timing)
    tf = time.time()
    if a.models_from:
        fx = Path(a.models_from).resolve()
        surv, mdir = fx / "survivors_planted.json", fx / "models_planted_lgbm" / "models" / "reinforcement"
    else:
        fx = out / "fx"
        fx.mkdir()
        surv, mdir = build_model(fx)
    timing["fixture_s"] = round(time.time() - tf, 1)
    worlds = []

    settings_before = settings_shas([])

    def world(tag, mdir_=None, **kw):
        r, w, v = make_world(out / "w" / tag, surv, mdir_ or mdir, **kw)
        worlds.append((r, w))
        settings_before.update(settings_shas([r]))
        return r, w, v
    snaps = []

    def snap_start(r, w):
        return (tree_state(r, skip=[r / "diagnostics_outputs"]), tree_state(w))

    def snap_ok(r, w, s):
        return tree_state(r, skip=[r / "diagnostics_outputs"]) == s[0] and tree_state(w) == s[1]

    # ---- success + parity -------------------------------------------------------------------
    if want("W1.success_attached") or want("W1.parity_hand_run") or want("W1.no_write_outside"):
        r, w, v1 = world("ok")
        s = snap_start(r, w)
        ts = time.time()
        res = W.run_for_flag(FID, [str(w)], root=str(r), out_root=str(out / "why_ok"), timeout_s=900,
                             votes_builder=copy_builder(v1), analysis=FAST)
        timing["runner_call_s"] = round(time.time() - ts, 1)
        timing["ll8_child"] = (res.get("children") or {}).get("ll8")
        g = wrf.read_flags(str(r))["groups"][FID]
        att = (g["attachments"] or [{}])[0]
        rr = json.load(open(Path(res["report_path"]).parent / "run_record.json")) if res.get("report_path") else {}
        ck_now = W.current_checkpoint(FAM, str(r))["checkpoint_sha256"]
        gate("W1.success_attached",
             res["status"] == "ok" and not g["open"] and g["status"] == "closed" and g["attempt_count"] == 1
             and g["attempts"][0]["outcome"] == "ok" and att.get("report_sha256") == sha(res["report_path"])
             and att.get("report_path") == res["report_path"]
             and rr.get("model_binding", {}).get("checkpoint_sha256") == ck_now
             and res["block"]["origin_summary"].endswith("origin_1/origin_summary.json"),
             f"status {res['status']} {res.get('reason')} {str(res.get('message'))[:200]}; flag {g.get('status')}; "
             f"attached sha == file sha: {att.get('report_sha256') == (sha(res['report_path']) if res.get('report_path') else None)}; "
             f"report checkpoint == model's: {rr.get('model_binding', {}).get('checkpoint_sha256') == ck_now}; "
             f"block {res.get('block', {}).get('origin_summary')}")
        if want("W1.parity_hand_run") and res.get("spec_path"):
            hd = out / "hand_ll8"
            p = subprocess.run([sys.executable, "-u", "-B", str(REPO / "scripts/diagnostics/attribution_cycle.py"),
                                "--spec", res["spec_path"], "--out-dir", str(hd), "--model-family", FAM,
                                "--flag-id", FID], capture_output=True, text=True,
                               env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"))
            cyc = Path(res["report_path"]).parent
            same = {n: sha(cyc / n) == sha(hd / n) for n in ("attribution_cycle.json", "extraction_priorities.json",
                                                             "signed_attribution.npz")} if p.returncode == 0 else {}
            DECLARED = ("argv", "seconds", "peak_rss_mib")
            r1, r2 = (json.load(open(d / "run_record.json")) for d in (cyc, hd)) if p.returncode == 0 else ({}, {1: 1})
            rr_same = {k: v for k, v in r1.items() if k not in DECLARED} == {k: v for k, v in r2.items() if k not in DECLARED}
            gate("W1.parity_hand_run", p.returncode == 0 and all(same.values()) and rr_same,
                 f"hand rc {p.returncode}; byte-identical {same}; run_record equal apart from declared {DECLARED}: {rr_same}")
        if want("W1.no_write_outside"):
            ok_tree = snap_ok(r, w, s)
            # negative control: a write next to the flags root's settings must be seen
            (r / "agent_manifests" / "stray.json").write_text("{}")
            ctl = not snap_ok(r, w, s)
            (r / "agent_manifests" / "stray.json").unlink()
            outs = sorted(str(p.relative_to(out / "why_ok")) for p in (out / "why_ok").rglob("*") if p.is_file())
            diag = sorted(p.name for p in (r / "diagnostics_outputs").iterdir())
            gate("W1.no_write_outside", ok_tree and ctl and diag == ["why_report_attachments.jsonl",
                                                                       "why_report_attempts.jsonl",
                                                                       "why_report_flags.jsonl"],
                 f"flags root (outside diagnostics_outputs) and wf root unchanged: {ok_tree}; negative control "
                 f"detected: {ctl}; diagnostics_outputs: {diag}; {len(outs)} files under the out root")

    # ---- second pass, change 1: V from the unlearned arm of the same pool size ---------------
    fake = [sys.executable, "-c", FAKE_LL8]
    if want("W1.votes_from_unlearned"):
        seen = {}

        def rec_builder(src):
            inner = copy_builder(src)

            def b(cand, out_, log, deadline):
                seen["votes_arm"] = dict(cand.get("votes_arm") or {})
                seen["grade_dirs"] = list(cand["grade_dirs"])
                return inner(cand, out_, log, deadline)
            return b
        r, w, v1 = world("unlearned", unlearned=True)
        res = W.run_for_flag(FID, [str(w)], root=str(r), out_root=str(out / "why_unl"), timeout_s=120,
                             votes_builder=rec_builder(v1), ll8_cmd=fake)
        g = wrf.read_flags(str(r))["groups"][FID]
        att = (g["attachments"] or [{}])[0]
        spec = json.load(open(res["spec_path"])) if res.get("spec_path") else {}
        vs = res.get("votes_source") or {}
        # the default builder writes its child spec from the same arm (the child then fails: synthetic arm)
        r2, w2, _ = world("unlearned_default", unlearned=True)
        res2 = W.run_for_flag(FID, [str(w2)], root=str(r2), out_root=str(out / "why_unl_default"), timeout_s=120)
        vspec = json.load(open(Path(res2["out_dir"]) / "votes_spec.json")) if res2.get("out_dir") and \
            (Path(res2["out_dir"]) / "votes_spec.json").is_file() else {}
        # fallback: no unlearned arm => the family arm, recorded as such
        r3, w3, v3 = world("no_unlearned")
        res3 = W.run_for_flag(FID, [str(w3)], root=str(r3), out_root=str(out / "why_unl_fb"), timeout_s=120,
                              votes_builder=copy_builder(v3), ll8_cmd=fake)
        fam_arm = f"F{FAM}s42_o1_p300"
        ok = (res["status"] == "ok" and vs.get("arm_id") == "A0_o1_p300" and vs.get("source") == "unlearned"
              and vs.get("grade_arm_id") == fam_arm and seen.get("votes_arm", {}).get("arm_id") == "A0_o1_p300"
              and all(f"/grade/{fam_arm}_" in d for d in seen.get("grade_dirs", ["x"]))
              and spec.get("votes", {}).get("source") == vs and att.get("votes_source") == vs
              and list(vspec.get("arms", {})) == ["A0_o1_p300"]
              and res3["status"] == "ok" and (res3.get("votes_source") or {}).get("arm_id") == fam_arm
              and (res3.get("votes_source") or {}).get("source") == "family_arm_fallback")
        gate("W1.votes_from_unlearned", ok,
             f"status {res['status']} {res.get('reason')}; votes_source {vs}; builder saw "
             f"{seen.get('votes_arm', {}).get('arm_id')}; grade dirs {seen.get('grade_dirs')}; spec source == result: "
             f"{spec.get('votes', {}).get('source') == vs}; attach source == result: {att.get('votes_source') == vs}; "
             f"default builder child spec arms {list(vspec.get('arms', {}))}; fallback {res3['status']} "
             f"{res3.get('votes_source')}")
    if want("W1.attach_sha_bound"):
        r, w, v1 = world("attach")
        res = W.run_for_flag(FID, [str(w)], root=str(r), out_root=str(out / "why_attach"), timeout_s=120,
                             votes_builder=copy_builder(v1), ll8_cmd=fake)
        g = wrf.read_flags(str(r))["groups"][FID]
        att = (g["attachments"] or [{}])[0]
        gate("W1.attach_sha_bound", res["status"] == "ok" and att.get("report_sha256") is not None
             and att.get("report_sha256") == sha(res["report_path"]),
             f"status {res['status']} {res.get('reason')}; attached sha {att.get('report_sha256')}")

    # ---- every failure reason -----------------------------------------------------------------
    def fail_case(reason, setup):
        name = f"W1.fail.{reason}"
        if not want(name):
            return
        r, w, v1 = world(reason)
        before = 0
        kw = dict(root=str(r), out_root=str(out / f"why_{reason}"), timeout_s=600, votes_builder=copy_builder(v1),
                  analysis=FAST)
        call = setup(r, w, v1, kw)
        res = call() if callable(call) else W.run_for_flag(FID, [str(w)], **kw)
        ok, det = failed_ok(wrf, r, FID, res, reason, before)
        gate(name, ok, det)

    fail_case("no_graded_block", lambda r, w, v, kw: (lambda: W.run_for_flag(FID, [str(w / "nothing_here")], **kw)))
    fail_case("no_checkpoint", lambda r, w, v, kw: shutil.rmtree(r / "models" / "reinforcement"))

    def _rm_surv(r, w, v, kw):
        (w / "origin_1" / "step3" / "survivors_with_scores.json").unlink()
    fail_case("missing_survivors", _rm_surv)

    def _bad_grade(r, w, v, kw):
        pd = next((w / "origin_1" / "grade").glob("*/per_draw.jsonl"))
        os.chmod(pd, 0o644)
        with open(pd, "a") as f:
            f.write("\n")
    fail_case("input_sha_mismatch", _bad_grade)

    def _shuffled(r, w, v, kw):
        kw["votes_builder"] = copy_builder(v, lambda d: dict(d, seeds=d["seeds"][::-1].copy()))
    fail_case("cycle_refused", _shuffled)

    def _no_V(r, w, v, kw):
        kw["votes_builder"] = copy_builder(v, lambda d: {k: x for k, x in d.items() if k != "V"})
    fail_case("cycle_error", _no_V)

    def _tiny(r, w, v, kw):
        kw["timeout_s"] = 1.0
    fail_case("timeout", _tiny)

    def _killed(r, w, v, kw):
        kw["ll8_cmd"] = [sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"]
    fail_case("killed", _killed)

    def _default_builder(r, w, v, kw):
        kw.pop("votes_builder")          # the real e2e_vote_npz child; the synthetic arm has no run record
    fail_case("spec_build_error", _default_builder)

    def _attach_err(r, w, v, kw):
        kw.update(ll8_cmd=fake, by=" ")  # attach refuses an empty `by`
    fail_case("attach_error", _attach_err)

    def _internal(r, w, v, kw):
        orig = W.current_checkpoint

        def call():
            W.current_checkpoint = lambda *x, **y: {}["boom"]
            try:
                return W.run_for_flag(FID, [str(w)], **kw)
            finally:
                W.current_checkpoint = orig
        return call
    fail_case("internal_error", _internal)

    def _no_draws(r, w, v, kw):          # [third pass] LL-8 ok, but no graded draw in any cell
        kw["ll8_cmd"] = [sys.executable, "-c", FAKE_LL8_NO_DRAWS]
    fail_case("no_graded_draws", _no_draws)

    # ---- one attempt per flag per run_open_flags call; failed listed first ------------------
    if want("W1.one_attempt_per_call") or want("W1.failed_listed_first"):
        r, w, v1 = world("multi", flags=((FID_OLD, "neural_net", "2026-11-25T01:00:00+00:00"),
                                         (FID, FAM, "2026-11-26T01:00:00+00:00"),
                                         ("catboost@2026-11-24T00:00:00+00:00", "catboost", "2026-11-24T00:00:00+00:00")))
        (r / "rep.md").write_text("r")
        wrf.attach("catboost@2026-11-24T00:00:00+00:00", "rep.md", "m", str(r))
        res1 = W.run_open_flags([str(w / "nothing_here")], root=str(r), out_root=str(out / "why_multi"))
        c1 = {k: g.get("attempt_count") for k, g in wrf.read_flags(str(r))["groups"].items()}
        res2 = W.run_open_flags([str(w / "nothing_here")], root=str(r), out_root=str(out / "why_multi"))
        c2 = {k: g.get("attempt_count") for k, g in wrf.read_flags(str(r))["groups"].items()}
        gate("W1.one_attempt_per_call",
             len(res1) == 2 and len(res2) == 2 and c1 == {FID_OLD: 1, FID: 1, "catboost@2026-11-24T00:00:00+00:00": 0}
             and c2 == {FID_OLD: 2, FID: 2, "catboost@2026-11-24T00:00:00+00:00": 0},
             f"after call 1 {c1}; after call 2 {c2}; results {len(res1)}, {len(res2)} "
             f"(reasons {[x.get('reason') for x in res1]})")
        # failed-first: a NEWER flag fails, an OLDER one has an ok attempt (still open, no report)
        r2, w2, _ = world("order", flags=((FID_OLD, "neural_net", "2026-11-25T01:00:00+00:00"),
                                          (FID, FAM, "2026-11-26T01:00:00+00:00")))
        wrf.record_attempt(FID_OLD, "ok", str(r2))
        W.run_for_flag(FID, [str(w2 / "nothing_here")], root=str(r2), out_root=str(out / "why_order"))
        lines = wrf.summary_lines(str(r2))
        gate("W1.failed_listed_first",
             len(lines) > 2 and lines[1].startswith(f"{wrf.TAG}   FAILED {FID}:") and "no_graded_block" in lines[1]
             and lines.index(next(l for l in lines if l.startswith(f"{wrf.TAG}   {FID_OLD}:"))) > 1,
             " | ".join(l[:110] for l in lines[:4]))

    # ---- nothing escapes run_open_flags ------------------------------------------------------
    if want("W1.no_exception_escapes"):
        r, w, v1 = world("escape")
        orig_run, orig_open = W.run_for_flag, W.wrf.open_flags
        logs.truncate(0)
        logs.seek(0)
        esc = []
        try:
            W.run_for_flag = lambda *x, **y: (_ for _ in ()).throw(RuntimeError("runner blew up"))
            res = W.run_open_flags([str(w)], root=str(r))
        except Exception as e:
            esc.append(repr(e))
            res = []
        finally:
            W.run_for_flag = orig_run
        g = wrf.read_flags(str(r))["groups"][FID]
        try:
            W.wrf.open_flags = lambda *x, **y: {}["x"]
            res_b = W.run_open_flags([str(w)], root=str(r))
        except Exception as e:
            esc.append(repr(e))
            res_b = None
        finally:
            W.wrf.open_flags = orig_open
        text = logs.getvalue()
        gate("W1.no_exception_escapes",
             not esc and len(res) == 1 and res[0]["reason"] == "internal_error" and g["status"] == "why_report_failed"
             and g["attempts"][-1]["reason"] == "internal_error" and "Traceback" in text
             and isinstance(res_b, list) and len(res_b) == 1 and res_b[0].get("status") == "failed"
             and res_b[0].get("reason") == "internal_error" and res_b[0].get("flag_id") is None
             and "ERROR why_report_runner" in text,
             f"escaped {esc}; result {[(x.get('reason'), x.get('message')) for x in res]}; flag {g.get('status')}; "
             f"reader failure -> {res_b}; ERROR with traceback logged: {'Traceback' in text}")

    # ---- CLI ---------------------------------------------------------------------------------
    if want("W1.cli"):
        r, w, v1 = world("cli")

        def cli(*args):
            p = subprocess.run([sys.executable, "-B", str(rf_path), *args], capture_output=True, text=True,
                               env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"))
            return p.returncode, p.stdout
        rcs = {"none": cli("--wf-root", str(w))[0],
               "both": cli("--flag-id", FID, "--all-open", "--wf-root", str(w))[0],
               "no_wf": cli("--flag-id", FID)[0],
               "unknown": cli("--flag-id", "nope@x", "--wf-root", str(w), "--root", str(r))[0],
               "bad_timeout": cli("--all-open", "--wf-root", str(w), "--root", str(r), "--timeout-s", "0")[0]}
        rc1, so = cli("--flag-id", FID, "--wf-root", str(w / "nothing_here"), "--root", str(r),
                      "--out-root", str(out / "why_cli"), "--json")
        js = json.loads(so) if so.strip().startswith("[") else []
        gate("W1.cli", all(v == 2 for v in rcs.values()) and rc1 == 1 and js and js[0]["reason"] == "no_graded_block",
             f"bad-argument exits {rcs}; failing flag exit {rc1}, json reason {js[0]['reason'] if js else None}")

    # ---- every except surfaces its error ----------------------------------------------------
    if want("W1.except_handlers_surface"):
        tree = ast.parse(rf_path.read_text())
        src = rf_path.read_text().split("\n")
        silent = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                body = "\n".join(src[node.body[0].lineno - 1:node.body[-1].end_lineno])
                typ = ast.unparse(node.type) if node.type else "bare"
                if typ == "ProcessLookupError":       # the child already exited: nothing to report
                    continue
                if not re.search(r"\braise\b|logger\.error|record_attempt|print\(.*ERROR|return 2", body):
                    silent.append(f"line {node.lineno} except {typ}")
        gate("W1.except_handlers_surface", not silent,
             f"handlers that neither raise, record, log at ERROR nor report: {silent or 'none'}")

    # ================= third pass (Beta X-W1-RUNNER R1) =========================================
    # ---- item 1 (Q1): the FAILED log append fails; the original reason is still recorded ----------
    if want("W1.log_append_failure"):
        orig_log = W._log
        paths = {}

        def bad_log(lp, msg):
            if msg.startswith("FAILED"):
                raise OSError(28, "No space left on device (injected)")
            return orig_log(lp, msg)
        res_by = {}
        W._log = bad_log
        try:
            for how in ("function", "cli", "all_open"):
                r, w, v1 = world(f"logfail_{how}")
                shutil.rmtree(r / "models" / "reinforcement")             # => no_checkpoint
                paths[how] = r
                err = io.StringIO()
                se = sys.stderr
                sys.stderr = err
                rc = None
                try:
                    if how == "function":
                        res = W.run_for_flag(FID, [str(w)], root=str(r), out_root=str(out / f"why_lf_{how}"))
                        rc = None
                    elif how == "cli":
                        so = sys.stdout
                        sys.stdout = io.StringIO()
                        try:
                            rc = W.main(["--flag-id", FID, "--wf-root", str(w), "--root", str(r), "--out-root",
                                         str(out / f"why_lf_{how}"), "--json"])
                            res = json.loads(sys.stdout.getvalue())[0]
                        finally:
                            sys.stdout = so
                    else:
                        res = W.run_open_flags([str(w)], root=str(r), out_root=str(out / f"why_lf_{how}"))[0]
                except Exception as e:           # an escape is a red gate, not a crashed suite
                    res = {"raised": repr(e)}
                finally:
                    sys.stderr = se
                g = wrf.read_flags(str(r))["groups"][FID]
                res_by[how] = (res, g, err.getvalue(), rc)
        finally:
            W._log = orig_log
        ok = all(res.get("reason") == "no_checkpoint" and res.get("log_error") and g["open"]
                 and g.get("attempt_count") == 1 and g["attempts"][-1]["reason"] == "no_checkpoint"
                 and "FAILED line not written" in e and "No space left" in e for res, g, e, _ in res_by.values())
        ok = ok and res_by["cli"][3] == 1
        gate("W1.log_append_failure", ok,
             "; ".join(f"{k}: raised {v[0].get('raised')} reason {v[0].get('reason')} recorded {(v[1].get('attempts') or [{}])[-1].get('reason')} "
                       f"log_error {bool(v[0].get('log_error'))} stderr {'FAILED line not written' in v[2]} rc {v[3]}"
                       for k, v in res_by.items()))

    # ---- item 2 (Q1): a child's leftover workers die whatever the ending ------------------------------
    if want("W1.child_group_cleanup"):
        cg = out / "child_group"
        cg.mkdir()
        CHILD = ("import subprocess, sys, os, signal, time\n"
                 "w = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
                 "open(sys.argv[1], 'w').write(f'{w.pid} {os.getpgid(0)}')\n"
                 "mode = sys.argv[2]\n"
                 "if mode == 'kill': os.kill(os.getpid(), signal.SIGKILL)\n"
                 "if mode == 'exit': sys.exit(0)\n"
                 "time.sleep(60)\n")

        def alive_in_group(pgid):
            out_ = []
            for d in Path("/proc").iterdir():
                if not d.name.isdigit():
                    continue
                try:
                    st = (d / "stat").read_text()
                except OSError:
                    continue
                f = st[st.rindex(")") + 2:].split()
                if int(f[2]) == pgid and f[0] != "Z":
                    out_.append(int(d.name))
            return out_
        det, okc = [], True
        for mode in ("kill", "exit", "interrupt"):
            pf = cg / f"worker_{mode}.pid"
            log_p = cg / f"{mode}.log"
            cmd = [sys.executable, "-c", CHILD, str(pf), mode]
            orig_sleep = W.time.sleep

            def sleeper(t, pf=pf):
                if pf.exists() and pf.read_text().strip():
                    raise KeyboardInterrupt("injected: wait interrupted")
                return orig_sleep(t)
            interrupted = False
            t0_ = time.monotonic()
            if mode == "interrupt":
                W.time.sleep = sleeper
            try:
                r_ = W.run_child(cmd, log_p, time.monotonic() + 20)
            except KeyboardInterrupt:
                interrupted, r_ = True, {}
            finally:
                W.time.sleep = orig_sleep
            wpid, pgid = (int(x) for x in pf.read_text().split()) if pf.exists() and pf.read_text().strip() \
                else (None, None)
            # the child's own group (start_new_session), which the worker shares: no live member may remain
            left = []
            for _ in range(50):
                left = alive_in_group(pgid) if pgid else []
                if not left:
                    break
                time.sleep(0.1)
            for x in left:          # scratch clean-up by recorded pid (never by pattern)
                try:
                    os.kill(x, 9)
                except ProcessLookupError:
                    pass
            good = wpid is not None and not left and (interrupted if mode == "interrupt" else True) \
                and time.monotonic() - t0_ < 15
            okc = okc and good
            det.append(f"{mode}: worker {wpid} left alive {left}, rc {r_.get('rc')}, interrupted {interrupted}")
        gate("W1.child_group_cleanup", okc, "; ".join(det))

    # ---- item 3 (Q3): a pooled-only origin gives its graded draws to the report -----------------------
    if want("W1.pooled_only"):
        r, w, v1 = world("pooled", pooled_only=True)
        res = W.run_for_flag(FID, [str(w)], root=str(r), out_root=str(out / "why_pooled"), timeout_s=900,
                             votes_builder=copy_builder(v1), analysis=FAST)
        g = wrf.read_flags(str(r))["groups"][FID]
        spec = json.load(open(res["spec_path"])) if res.get("spec_path") else {}
        rep = json.load(open(res["report_path"])) if res.get("report_path") else {}
        draws = {k: c.get("draws") for k, c in (rep.get("contrasts") or {}).items()}
        gate("W1.pooled_only",
             res["status"] == "ok" and not g["open"] and sorted(spec.get("sessions") or []) == DATASET_SESSIONS
             and sum(draws.values()) > 0 and all(x.endswith("_pooled") for x in spec.get("grade_dirs", ["x"]))
             and res.get("graded_draws_in_cells") == sum(draws.values()),
             f"status {res['status']} {res.get('reason')} {str(res.get('message'))[:160]}; spec sessions "
             f"{spec.get('sessions')}; cells {draws}")

    # ---- item 4 (Q3): analysis may set only the declared analysis keys --------------------------------
    if want("W1.analysis_keys"):
        r, w, v1 = world("analysis")
        called = []

        def spy_builder(*x):
            called.append(1)
            return copy_builder(v1)(*x)
        det, okc = [], True
        for i, bad in enumerate(({"blocks": {"elsewhere": [0, 2]}}, {"grade_dirs": []}, {"sessions": []},
                                 {"survivors": {}}, {"models_dir": "x"}, {"bootstrap": {"B": 400}, "pool_sizes": [1]})):
            od = out / f"why_analysis_{i}"
            res = W.run_for_flag(FID, [str(w)], root=str(r), out_root=str(od), timeout_s=120,
                                 votes_builder=spy_builder, ll8_cmd=fake, analysis=bad)
            good = res["status"] == "failed" and res["reason"] == "spec_build_error" and not od.exists()
            okc = okc and good
            det.append(f"{list(bad)}: {res['status']}/{res['reason']} out created {od.exists()}")
        g = wrf.read_flags(str(r))["groups"][FID]
        rc = subprocess.run([sys.executable, "-B", str(rf_path), "--flag-id", FID, "--wf-root", str(w), "--root",
                             str(r), "--out-root", str(out / "why_analysis_cli"), "--analysis-json",
                             json.dumps({"blocks": {"x": [0, 2]}})], capture_output=True, text=True,
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")).returncode
        # an allowed key still runs (fake LL-8)
        res_ok = W.run_for_flag(FID, [str(w)], root=str(r), out_root=str(out / "why_analysis_ok"), timeout_s=120,
                                votes_builder=copy_builder(v1), ll8_cmd=fake, analysis={"seed": 5, **FAST})
        spec_ok = json.load(open(res_ok["spec_path"])) if res_ok.get("spec_path") else {}
        gate("W1.analysis_keys", okc and not called and g["open"] and rc == 2 and res_ok["status"] == "ok"
             and spec_ok.get("seed") == 5 and spec_ok.get("bootstrap") == {"B": 400},
             f"{'; '.join(det)}; builder called {len(called)}x; flag open {g['open']}; CLI rc {rc}; allowed keys "
             f"-> {res_ok['status']} seed {spec_ok.get('seed')}")

    # ---- item 5 (Q3): each grade dir is bound to the origin-pinned grader summary -----------------------
    if want("W1.grade_binding"):
        det, okc = [], True

        def tamper(tag, fn):
            nonlocal okc
            r, w, v1 = world(f"bind_{tag}")
            o1 = w / "origin_1"
            s1 = json.load(open(o1 / "origin_summary.json"))
            gd = Path(s1["grades"][0]["dir"])
            fn(o1, s1, gd, w)
            res = W.run_for_flag(FID, [str(w)], root=str(r), out_root=str(out / f"why_bind_{tag}"), timeout_s=120,
                                 votes_builder=copy_builder(v1), ll8_cmd=fake)
            g = wrf.read_flags(str(r))["groups"][FID]
            good = res["status"] == "failed" and res["reason"] == "input_sha_mismatch" and g["open"]
            okc = okc and good
            det.append(f"{tag}: {res['status']}/{res['reason']}")

        def resummary(o1, s1, gd, edit):     # a consistent pin: only the binding differs
            gs = json.load(open(gd / "summary.json"))
            edit(gs)
            os.chmod(gd / "summary.json", 0o644)
            write_json(gd / "summary.json", gs)
            s1["grades"][0]["summary_sha256"] = sha(gd / "summary.json")
            write_json(o1 / "origin_summary.json", s1)
        tamper("missing_summary", lambda o1, s1, gd, w: (gd / "summary.json").unlink())
        def _pin(o1, s1, gd, w):
            os.chmod(gd / "summary.json", 0o644)
            with open(gd / "summary.json", "a") as f:
                f.write(" ")
        tamper("pin_differs", _pin)
        tamper("other_arm_pools", lambda o1, s1, gd, w: resummary(
            o1, s1, gd, lambda gs: gs["inputs"]["pools"].update(sha256="e" * 64)))
        tamper("other_dataset", lambda o1, s1, gd, w: resummary(
            o1, s1, gd, lambda gs: gs["inputs"]["dataset"].update(sha256="f" * 64)))
        tamper("other_block", lambda o1, s1, gd, w: resummary(
            o1, s1, gd, lambda gs: gs["grade_block"].update(start_index=1)))
        tamper("other_pool_size", lambda o1, s1, gd, w: resummary(
            o1, s1, gd, lambda gs: gs["declared"]["arms"][0].update(max_k=20)))
        tamper("other_sessions", lambda o1, s1, gd, w: resummary(
            o1, s1, gd, lambda gs: gs["grade_block"].update(sessions=["evening"])))
        tamper("per_draw_not_summarised", lambda o1, s1, gd, w: resummary(
            o1, s1, gd, lambda gs: gs["outputs"]["per_draw.jsonl"].update(sha256="0" * 64)))
        gate("W1.grade_binding", okc, "; ".join(det))

    # ---- item 6 (Q4): out root inside <tree>/data/, checked before anything is created ------------------
    if want("W1.out_root_boundary"):
        r, w, v1 = world("outroot")
        outside = [REPO / "w1_outside_probe_dir", REPO / "data" / ".." / "w1_outside_probe_dir2"]
        det, okc = [], True
        for od in outside:
            pre = od.resolve().exists()
            res = W.run_for_flag(FID, [str(w)], root=str(r), out_root=str(od), timeout_s=60,
                                 votes_builder=copy_builder(v1), ll8_cmd=fake)
            rc = subprocess.run([sys.executable, "-B", str(rf_path), "--flag-id", FID, "--wf-root", str(w),
                                 "--root", str(r), "--out-root", str(od)], capture_output=True, text=True,
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")).returncode
            good = (not pre and not od.resolve().exists() and res["reason"] == "spec_build_error"
                    and res.get("out_dir") is None and rc == 2)
            okc = okc and good
            det.append(f"{od.resolve()}: {res['status']}/{res['reason']} created {od.resolve().exists()} CLI rc {rc}")
        gate("W1.out_root_boundary", okc, "; ".join(det))

    # ---- item 6 (Q4): the documented plain CLI writes no bytecode (clean scratch copy) -----------------
    if want("W1.cli_no_bytecode"):
        st = out / "scratch_tree"
        (st / "scripts" / "diagnostics").mkdir(parents=True)
        shutil.copy(REPO / RUNNER_REL, st / RUNNER_REL)
        shutil.copy(REPO / "why_report_flags.py", st / "why_report_flags.py")
        shutil.copytree(REPO / "models", st / "models", ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", "reinforcement", "test_*", "*(bak*"))
        (st / "data").mkdir()
        r, w, v1 = world("bytecode")
        fr_ = st / "data" / "root"
        shutil.copytree(r, fr_)
        env = {k: v for k, v in os.environ.items() if k != "PYTHONDONTWRITEBYTECODE"}
        env["CUDA_VISIBLE_DEVICES"] = ""
        pyc_before = sorted(str(p_) for p_ in st.rglob("*") if p_.name == "__pycache__" or p_.suffix == ".pyc")
        p_ = subprocess.run([sys.executable, str(st / RUNNER_REL), "--flag-id", FID, "--wf-root",
                             str(w / "nothing_here"), "--root", str(fr_), "--models-dir",
                             str(fr_ / "models" / "reinforcement"), "--json"],
                            capture_output=True, text=True, env=env, cwd=str(st))
        js = json.loads(p_.stdout) if p_.stdout.strip().startswith("[") else [{}]
        imported_models = js[0].get("checkpoint") is not None      # current_checkpoint ran: models.* imported
        pyc = sorted(str(x.relative_to(st)) for x in st.rglob("*") if x.name == "__pycache__" or x.suffix == ".pyc")
        # negative control: the same environment DOES write bytecode for a plain import of the copy
        ctl_dir = out / "scratch_tree_ctl"
        shutil.copytree(st / "models", ctl_dir / "models", ignore=shutil.ignore_patterns("__pycache__"))
        subprocess.run([sys.executable, "-c", "import models.inference_bundle"], cwd=str(ctl_dir), env=env,
                       capture_output=True)
        ctl = any(x.name == "__pycache__" for x in ctl_dir.rglob("*"))
        gate("W1.cli_no_bytecode",
             p_.returncode == 1 and js[0].get("reason") == "no_graded_block" and imported_models and not pyc
             and not pyc_before and ctl,
             f"plain `python3 {RUNNER_REL}` (no -B, no PYTHONDONTWRITEBYTECODE) rc {p_.returncode} reason "
             f"{js[0].get('reason')}; models imported {imported_models}; bytecode in the scratch tree: {pyc or 'none'}; "
             f"negative control (plain import, same env) wrote __pycache__: {ctl}; stderr {p_.stderr[-200:]!r}")

    # ---- item 7 (Q2): an unreadable attempts file hides no flag ------------------------------------------
    if want("W1.attempts_unreadable"):
        r, w, v1 = world("attempts_ro", flags=((FID_OLD, "neural_net", "2026-11-25T01:00:00+00:00"),
                                               (FID, FAM, "2026-11-26T01:00:00+00:00")))
        (r / "rep.md").write_text("r")
        wrf.attach(FID_OLD, "rep.md", "m", str(r))
        wrf.record_attempt(FID, "failed", str(r), reason="timeout", error_class="X", message="m")
        ap = r / wrf.ATTEMPTS_PATH
        wlogs = io.StringIO()
        wh = logging.StreamHandler(wlogs)
        wh.setLevel(logging.ERROR)
        logging.getLogger("why_report_flags").addHandler(wh)
        os.chmod(ap, 0o000)
        try:
            rf_ = wrf.read_flags(str(r))
            lines = wrf.summary_lines(str(r))
            buf = io.StringIO()
            so = sys.stdout
            sys.stdout = buf
            try:
                lrc = wrf.main(["--root", str(r), "list", "--json"])
            finally:
                sys.stdout = so
            lj = json.loads(buf.getvalue())
            res_open = W.run_open_flags([str(w)], root=str(r), out_root=str(out / "why_attempts_ro"))
            cp = subprocess.run([sys.executable, "-B", str(rf_path), "--all-open", "--wf-root", str(w), "--root",
                                 str(r), "--out-root", str(out / "why_attempts_ro_cli")], capture_output=True,
                                text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"))
        finally:
            os.chmod(ap, 0o644)
            logging.getLogger("why_report_flags").removeHandler(wh)
        g = rf_["groups"]
        ok = (FID in g and g[FID]["open"] and not g[FID_OLD]["open"] and g[FID_OLD]["attachments"]
              and rf_.get("attempts_error") and any(f"   {FID}:" in l for l in lines)
              and any("attempts file unreadable" in l for l in lines) and lrc == 0
              and [x["flag_id"] for x in lj["flags"]] == [FID] and lj.get("attempts_error")
              and "unreadable" in wlogs.getvalue()
              and len(res_open) == 1 and res_open[0]["status"] == "failed" and res_open[0].get("attempt_record_error")
              and cp.returncode == 1)
        gate("W1.attempts_unreadable", ok,
             f"flags {list(g)} open {[x['open'] for x in g.values()]}; attempts_error {rf_.get('attempts_error')}; "
             f"summary {[l[:70] for l in lines]}; list --json flags {[x['flag_id'] for x in lj['flags']]}; "
             f"runner {[(x['status'], x.get('reason'), bool(x.get('attempt_record_error'))) for x in res_open]}; "
             f"CLI --all-open rc {cp.returncode}")

    # ---- item 8 (Q6): LL-8's G-Q4-SCALER refusal carried through the runner ----------------------------
    if want("W1.g_q4_scaler"):
        import test_ll8_attribution_cycle as ll8
        nn_dir = fx / "models_nn" / "models" / "reinforcement"
        if not nn_dir.is_dir():
            train_nn(fx, surv)
        rel = ll8.relocate_nn(fx, nn_dir)          # the written checkpoint_path now points at nothing
        fid_nn = "neural_net@2026-11-26T03:00:00+00:00"
        r, w, v1 = world("gq4", flags=((fid_nn, "neural_net", "2026-11-26T03:00:00+00:00"),), fam="neural_net",
                         mdir_=rel)
        res = W.run_for_flag(fid_nn, [str(w)], root=str(r), out_root=str(out / "why_gq4"), timeout_s=900,
                             votes_builder=copy_builder(v1), analysis=FAST)
        g = wrf.read_flags(str(r))["groups"][fid_nn]
        t = (g.get("attempts") or [{}])[-1]
        gate("W1.g_q4_scaler",
             res["status"] == "failed" and res["reason"] == "cycle_refused" and "G-Q4-SCALER" in (res["message"] or "")
             and g["open"] and not g["attachments"] and t.get("reason") == "cycle_refused"
             and "G-Q4-SCALER" in (t.get("message") or ""),
             f"{res['status']}/{res['reason']}: {str(res.get('message'))[:200]}; flag open {g['open']}; recorded "
             f"{t.get('reason')}")

    # ---- settings -----------------------------------------------------------------------------
    if want("W1.settings_unchanged"):
        if not worlds:
            r, w, v1 = world("settings")
            W.run_for_flag(FID, [str(w / "nothing_here")], root=str(r), out_root=str(out / "why_settings"))
        after = settings_shas([x[0] for x in worlds])
        before_cmp = settings_before
        gate("W1.settings_unchanged", before_cmp == after and len(after) >= 4,
             f"{len(after)} files (watcher_policies.json, agent_manifests/*.json in the tree and the flags root); "
             f"changed: {[k for k in after if before_cmp.get(k) != after[k]] or 'none'}")

    # the tree itself (outside --out): nothing written
    repo_after = ll6.boundary_snapshot(boundary)
    if only is None:
        dif = ll6.boundary_diff(repo_before, repo_after)
        gate("W1.tree_unchanged_outside_out", repo_before == repo_after,
             f"git status outside data/ and every path under data/ except {boundary}, before vs after; diff {str(dif)[:300]}")
    mut = []
    if not a.no_mutants and only is None:
        mut = run_mutants(out, fx)
    return finish(out, t0, mut, timing)


def run_mutants(out, fx):
    green = {r["gate"]: r["ok"] for r in RESULTS}
    res = []
    src = (REPO / RUNNER_REL).read_text()
    for mid, reps, targets in MUTANTS:
        root = out / "mutants" / mid
        if not all(green.get(t) is True for t in targets):
            res.append({"id": mid, "killed": False, "detail": f"target not green unmutated: {targets}"})
            continue
        s = src
        bad = [o[:60] for o, _ in reps if s.count(o) != 1]
        if bad:
            res.append({"id": mid, "killed": False, "detail": f"anchor not found exactly once: {bad}"})
            continue
        for o, n in reps:
            ind = o[:len(o) - len(o.lstrip())]
            s = s.replace(o, ind + MARK + "\n" + n)
        f = root / "code" / RUNNER_REL
        f.parent.mkdir(parents=True)
        f.write_text(s)
        mark = root / "mark"
        p = subprocess.run([sys.executable, "-u", "-B", str(Path(__file__).resolve()), "--out", str(root / "run"),
                            "--runner-file", str(f), "--models-from", str(fx), "--no-mutants",
                            "--only", ",".join(targets)], capture_output=True, text=True,
                           env=dict(os.environ, W1_MUTANT_MARK=str(mark), CUDA_VISIBLE_DEVICES="",
                                    PYTHONDONTWRITEBYTECODE="1"))
        (root / "run.log").write_text(p.stdout + p.stderr)
        try:
            g = {r["gate"]: r["ok"] for r in json.load(open(root / "run" / "selftest_summary.json"))["gates"]}
        except Exception as e:
            res.append({"id": mid, "killed": False, "detail": f"no summary ({e}); rc {p.returncode}"})
            continue
        red = [t for t in targets if g.get(t) is False]
        killed = len(red) == len(targets) and mark.exists()
        res.append({"id": mid, "killed": killed, "targets_red": red, "marker": "hit" if mark.exists() else "absent",
                    "applied": f"{len(reps)} edit(s), each anchor found once; sha {sha(REPO / RUNNER_REL)[:12]} -> {sha(f)[:12]}"})
    for r in res:
        print(f"[{'PASS' if r['killed'] else 'FAIL'}] MUTANT W1.{r['id']} killed={r['killed']} "
              f"red={r.get('targets_red')} marker={r.get('marker')} {r.get('applied', r.get('detail'))}", flush=True)
    return res


def finish(out, t0, mut, timing):
    import resource
    ng, npass = len(RESULTS), sum(r["ok"] for r in RESULTS)
    nm, nk = len(mut), sum(m["killed"] for m in mut)
    summ = {"gates": RESULTS, "mutants": mut, "timing": timing, "seconds": round(time.time() - t0, 1),
            "peak_rss_mib_self": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "peak_rss_mib_largest_child": round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024, 1)}
    (out / "selftest_summary.json").write_text(json.dumps(summ, indent=1, default=str))
    print(f"gates {npass}/{ng}; mutants {nk}/{nm}; {summ['seconds']} s; timing {json.dumps(timing, default=str)}",
          flush=True)
    full = npass == ng and nm == len(MUTANTS) and nk == nm
    if full:
        print(f"COMPLETION SENTINEL: PASS ({npass}/{ng} gates, {nk}/{nm} mutants killed)", flush=True)
    return 0 if full or (nm == 0 and npass == ng) else 1


if __name__ == "__main__":
    sys.exit(main())
