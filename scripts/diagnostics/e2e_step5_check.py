#!/usr/bin/env python3
"""
e2e_step5_check.py - NON-CERTIFYING DIAGNOSTIC. Read-only checks of one finished
Step 5 family run dir made by data/e2e_runH/step45/run_one.sh. Writes one JSON
report to --out (must not exist). Loads the checkpoint in this process on CPU
(run with CUDA_VISIBLE_DEVICES="").

Checks: shim exit and child failures; first-permutation sha256; sidecar fresh and
complete; Optuna study db (read-only, immutable) trial states and best value equal
to the sidecar; checkpoint loads and its feature count equals the sidecar's;
artifact sha256s; for the NN, the CAT-B and S121 lines in every captured child
stderr and the checkpoint normalisation keys.
"""
import argparse
import glob
import hashlib
import json
import math
import os
import sqlite3
import sys


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_checkpoint(fam, path):
    if fam == "neural_net":
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
        keys = sorted(ck)
        info = {"keys": keys, "feature_count": int(ck.get("feature_count", -1)),
                "hidden_layers": ck.get("hidden_layers"), "dropout": ck.get("dropout"),
                "best_epoch": ck.get("best_epoch"),
                "normalize_features": ck.get("normalize_features"), "use_leaky_relu": ck.get("use_leaky_relu"),
                "y_mean": ck.get("y_mean"), "y_std": ck.get("y_std")}
        for k in ("scaler_mean", "scaler_scale"):
            v = ck.get(k)
            info[k] = None if v is None else {"shape": list(getattr(v, "shape", [])),
                                               "finite": bool(all(math.isfinite(float(x)) for x in v)),
                                               "min": float(min(v)), "max": float(max(v))}
        info["has_required_norm_keys"] = all(k in ck for k in ("scaler_mean", "scaler_scale", "y_mean", "y_std"))
        return info
    if fam == "lightgbm":
        import lightgbm as lgb
        b = lgb.Booster(model_file=path)
        return {"feature_count": int(b.num_feature()), "num_trees": int(b.num_trees())}
    if fam == "xgboost":
        import xgboost as xgb
        b = xgb.Booster()
        b.load_model(path)
        return {"feature_count": int(b.num_features()), "num_boosted_rounds": int(b.num_boosted_rounds())}
    if fam == "catboost":
        from catboost import CatBoost
        m = CatBoost()
        m.load_model(path)
        p = m.get_all_params()
        return {"feature_count": len(m.feature_names_), "tree_count": int(m.tree_count_),
                "depth": p.get("depth"), "iterations": p.get("iterations"), "task_type": p.get("task_type")}
    raise ValueError(fam)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--family", required=True, choices=["neural_net", "lightgbm", "xgboost", "catboost"])
    ap.add_argument("--expected-trials", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit(f"refuse: {a.out} exists")
    R = os.path.abspath(a.run_dir)
    rep = {"kind": "e2e_step5_check", "label": "NON-CERTIFYING DIAGNOSTIC", "run_dir": R, "family": a.family}
    problems = []

    shim = json.load(open(os.path.join(R, "shim_capture", "shim_summary.json")))
    rep["shim"] = {k: shim.get(k) for k in ("exit_code", "wall_seconds", "child_calls", "child_failures",
                                            "child_scripts", "first_permutation", "split_seed", "script_sha256", "argv")}
    if shim.get("exit_code") != 0:
        problems.append("shim exit_code != 0")
    if shim.get("child_failures"):
        problems.append("child call failures")

    env = dict(l.rstrip("\n").split("=", 1) for l in open(os.path.join(R, "run_env.txt")) if "=" in l)
    rep["run_env"] = env

    md = os.path.join(R, "models", "reinforcement")
    sc_path = os.path.join(md, "best_model.meta.json")
    sc = json.load(open(sc_path))
    rep["sidecar"] = {"path": sc_path, "sha256": sha(sc_path), "mtime": os.path.getmtime(sc_path),
                      "model_type": sc.get("model_type"), "checkpoint_path": sc.get("checkpoint_path"),
                      "checkpoint_format": sc.get("checkpoint_format"),
                      "feature_count": sc.get("feature_schema", {}).get("feature_count"),
                      "feature_schema_hash": sc.get("feature_schema", {}).get("feature_schema_hash"),
                      "optuna": sc.get("optuna"), "training_metrics": sc.get("training_metrics"),
                      "hyperparameters": sc.get("hyperparameters"), "signal_quality": sc.get("signal_quality"),
                      "data_context": sc.get("data_context"), "hardware": sc.get("hardware"),
                      "provenance": sc.get("provenance"), "training_info": sc.get("training_info")}
    if sc.get("model_type") != a.family:
        problems.append("sidecar model_type mismatch")
    if not os.path.abspath(sc.get("checkpoint_path", "")).startswith(R + os.sep):
        problems.append("sidecar checkpoint_path outside run dir")
    # The NN sidecar carries optuna.best_value (meta_prediction_optimizer_anti_overfit.py:2908).
    # Tree sidecars are written by MultiModelTrainer.save_model (:1216) with no optuna block, so for
    # them the CV score is taken from the study db and cross-checked against the log's "Best R²" line.
    bv = (sc.get("optuna") or {}).get("best_value")
    rep["cv_score_source"] = "sidecar optuna.best_value" if bv is not None else "study db (sidecar has no optuna block)"
    if "optuna" not in sc:
        rep["sidecar_has_optuna_block"] = False

    dbs = glob.glob(os.path.join(R, "optuna_studies", "*.db"))
    rep["study_dbs"] = []
    for db in dbs:
        con = sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)
        names = [r[0] for r in con.execute("select study_name from studies")]
        states = dict(con.execute("select state, count(*) from trials group by state").fetchall())
        vals = [r[0] for r in con.execute(
            "select v.value from trial_values v join trials t on t.trial_id=v.trial_id where t.state='COMPLETE'")]
        con.close()
        rep["study_dbs"].append({"path": db, "sha256": sha(db), "studies": names, "trial_states": states,
                                 "complete_values_n": len(vals),
                                 "max_value": max(vals) if vals else None,
                                 "min_value": min(vals) if vals else None,
                                 "all_finite": all(math.isfinite(v) for v in vals)})
    if len(dbs) != 1:
        problems.append(f"{len(dbs)} study dbs (expected 1, fresh)")
    else:
        d = rep["study_dbs"][0]
        if d["trial_states"].get("COMPLETE") != a.expected_trials or len(d["trial_states"]) != 1:
            problems.append(f"trial states {d['trial_states']} != {a.expected_trials} COMPLETE")
        if bv is None:
            bv = d["max_value"]
        elif d["max_value"] is None or abs(d["max_value"] - bv) > 1e-12:
            problems.append("db best value != sidecar best_value")
    log = os.path.join(os.path.dirname(R), os.path.basename(R) + ".log")
    best_lines = [l.strip() for l in open(log) if "Best R²:" in l] if os.path.exists(log) else []
    rep["log_best_lines"] = best_lines
    if bv is None or not math.isfinite(bv):
        problems.append("CV score not finite / missing")
    elif not best_lines or f"{bv:.6f}" not in best_lines[-1]:
        problems.append("log 'Best R²' line does not match the CV score")
    rep["cv_score"] = bv

    ck = sc.get("checkpoint_path")
    try:
        rep["checkpoint"] = {"path": ck, "sha256": sha(ck), "bytes": os.path.getsize(ck),
                             "loaded": load_checkpoint(a.family, ck)}
        if rep["checkpoint"]["loaded"]["feature_count"] != rep["sidecar"]["feature_count"]:
            problems.append("checkpoint feature_count != sidecar")
    except Exception as e:
        rep["checkpoint"] = {"path": ck, "error": repr(e)}
        problems.append("checkpoint failed to load")

    rep["artifacts_sha256"] = {os.path.relpath(p, R): sha(p) for p in sorted(
        glob.glob(os.path.join(md, "*")) + glob.glob(os.path.join(R, "diagnostics_outputs", "*")))
        if os.path.isfile(p)}
    rep["leftover_tmp"] = sorted(os.path.relpath(p, R) for p in glob.glob(os.path.join(md, "tmp", "**"), recursive=True)
                                 if os.path.isfile(p))

    if a.family == "neural_net":
        calls = [json.loads(l) for l in open(os.path.join(R, "shim_capture", "calls.jsonl"))]
        tst = [c for c in calls if c["argv"][1:2] == ["train_single_trial.py"]]
        def lines(c, pat):
            f = c.get("stderr_file")
            if not f:
                return []
            return [l.rstrip("\n") for l in open(os.path.join(R, "shim_capture", f)) if pat in l]
        final = [c for c in tst if "--save-model" in c["argv"]]
        rep["nn_normalisation"] = {
            "train_single_trial_calls": len(tst),
            "fold_calls": len(tst) - len(final), "final_calls": len(final),
            "calls_with_CAT-B_line": sum(1 for c in tst if lines(c, "[CAT-B] Input normalization applied")),
            "calls_with_S121_line": sum(1 for c in tst if lines(c, "[S121] Y normalization")),
            "final_call_lines": (lines(final[-1], "[CAT-B] Input normalization applied")
                                 + lines(final[-1], "[S121] Y normalization")) if final else [],
            "final_call_device_line": (lines(final[-1], "device") or [])[:3] if final else [],
            "final_call_argv": final[-1]["argv"] if final else None,
            "final_call_cwd": final[-1]["cwd"] if final else None,
            "scripts_used": sorted({(c["script_resolved"], c["script_sha256"]) for c in tst}),
            "in_step5_log": "lines are printed to the child's stderr (train_single_trial.py:493-495, :510-512), "
                            "which the parent captures and drops on success (meta_prediction_optimizer_anti_overfit.py:1949-1983, :2602)",
        }
        nn = rep["nn_normalisation"]
        if nn["calls_with_CAT-B_line"] != len(tst) or nn["calls_with_S121_line"] != len(tst):
            problems.append("CAT-B/S121 line missing in some child call")
        if not rep.get("checkpoint", {}).get("loaded", {}).get("has_required_norm_keys"):
            problems.append("checkpoint missing scaler_mean/scaler_scale/y_mean/y_std")

    rep["problems"] = problems
    rep["status"] = "ok" if not problems else "problems"
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=1, default=str)
    print(json.dumps({"family": a.family, "status": rep["status"], "problems": problems,
                      "best_value": bv, "out": a.out}))


if __name__ == "__main__":
    main()
