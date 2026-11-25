#!/usr/bin/env python3
"""
e2e_step5_shim.py - NON-CERTIFYING DIAGNOSTIC. Runs one production Step 5 CLI call
(meta_prediction_optimizer_anti_overfit.py) unchanged, as __main__, with two
observation/declaration additions and nothing else:

1. --split-seed N seeds numpy's global RNG once before the module runs.
   Why: the module's 80/20 train+val/test split is an unseeded
   np.random.permutation (meta_prediction_optimizer_anti_overfit.py:1514;
   already recorded, docs/CHAPTER_6_AUDIT.md:140). The KFold folds are seeded
   (random_state=42, :2034) but are drawn from that unseeded train+val set, so
   four separate family calls would otherwise CV on four different survivor
   sets. The declared winner rule needs the same folds for every family.
   The first permutation's sha256 is recorded so equality across families is
   checked, not assumed.  (TODO-E2E-21: seed the split in the module as a
   declared, recorded parameter; until then this shim carries it.)

2. subprocess.run in the parent is wrapped to RECORD each child call (argv,
   cwd, resolved script path, return code, seconds, stdout and stderr) to
   --capture-dir. The child's result is returned to the module unchanged.
   Why: the module captures train_single_trial.py's stderr and discards it on
   success (:1949-1983 folds, :2602 final), so the lines "[CAT-B] Input
   normalization applied" and "[S121] Y normalization" (train_single_trial.py:
   493, 510) never reach any log. Also records which train_single_trial.py ran
   (TODO-E2E-7) and every fold failure, which the module turns into r2=-999.0
   and averages in silently (:1957, :2118). (TODO-E2E-22.)

3. --target-field F (TASK_LOOP_WF, 2026-11-23): the Step 5 training target as a
   declared setting. Default `holdout_quality` = today: nothing is rewritten and
   the module's argv is passed through unchanged (same path as before this
   setting existed). The module hard-codes its target
   (meta_prediction_optimizer_anti_overfit.py:1479-1483, target_field=
   'holdout_quality'; loader :612-689 takes features[F] first, then the top-level
   item[F], else 0.0 silently). For any other F the shim writes a DERIVED
   survivors file into --capture-dir in which each row's top-level
   `holdout_quality` is replaced by that row's F value, and points the module's
   --survivors at it. Nothing else in a row changes; no row is dropped.
   Allowed F: `holdout_hits`, or `holdout_features.<name>` (the labels Step 3
   writes: full_scoring_worker.py top-level holdout_hits / holdout_features).
   These lie outside row['features'], which is the only source of X
   (get_feature_schema_from_data :546-570; loader :649-650), so the target can
   never also be a training column. Fail-closed: F missing, non-numeric or
   non-finite on any row; `features` holding a key named holdout_quality (the
   loader would read it first, :655); the derived file not changing y as
   declared (re-read check). The original and derived sha256, the row count and
   y statistics are in shim_summary.json `target`. (TODO-E2E-25: carry the
   target in the module as a declared flag; until then this shim carries it.)

No module code is changed. With the default target the shim does not alter
args, env, or results.
"""
import argparse
import hashlib
import json
import math
import os
import runpy
import subprocess
import sys
import time

import numpy as np

DEFAULT_TARGET = "holdout_quality"
MODULE_TARGET_KEY = "holdout_quality"   # meta_prediction_optimizer_anti_overfit.py:1481


def _sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _target_value(row, field):
    if field == "holdout_hits":
        return row.get("holdout_hits")
    if field.startswith("holdout_features."):
        hf = row.get("holdout_features")
        return hf.get(field.split(".", 1)[1]) if isinstance(hf, dict) else None
    raise SystemExit(f"[E2E-SHIM] refuse: --target-field {field!r} not allowed "
                     f"(holdout_quality, holdout_hits, holdout_features.<name>)")


def derive_target_file(src, dst, field):
    """Write a copy of the survivors file with top-level holdout_quality := row[field]."""
    rows = json.load(open(src))
    if not isinstance(rows, list) or not rows:
        raise SystemExit("[E2E-SHIM] refuse: survivors file is not a non-empty list")
    ys = []
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            raise SystemExit(f"[E2E-SHIM] refuse: row {i} is not an object")
        if MODULE_TARGET_KEY in (r.get("features") or {}):
            raise SystemExit(f"[E2E-SHIM] refuse: row {i} features hold '{MODULE_TARGET_KEY}' "
                             f"(the loader would read it before the top-level value, :655)")
        v = _target_value(r, field)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            raise SystemExit(f"[E2E-SHIM] refuse: row {i} {field} = {v!r} (missing / non-numeric / "
                             f"non-finite; the module would silently use 0.0, :658-660)")
        r[MODULE_TARGET_KEY] = v
        ys.append(float(v))
    with open(dst + ".partial", "w") as f:
        json.dump(rows, f)
    os.replace(dst + ".partial", dst)
    back = [float(r[MODULE_TARGET_KEY]) for r in json.load(open(dst))]
    if back != ys:
        raise SystemExit("[E2E-SHIM] refuse: derived file re-read does not give the declared y")
    y = np.asarray(ys, dtype=np.float64)
    return {"rows": len(ys), "y_mean": float(y.mean()), "y_std": float(y.std()),
            "y_min": float(y.min()), "y_max": float(y.max()),
            "y_distinct": int(len(np.unique(y)))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 allow_abbrev=False)   # no silent prefix matching (ADDENDUM_4 Fix 2)
    ap.add_argument("--split-seed", type=int, required=True)
    ap.add_argument("--capture-dir", required=True)
    ap.add_argument("--script", required=True, help="path to meta_prediction_optimizer_anti_overfit.py")
    ap.add_argument("--target-field", default=DEFAULT_TARGET,
                    help="Step 5 training target (default holdout_quality = today, no rewrite); "
                         "or holdout_hits, holdout_features.<name>")
    ap.add_argument("rest", nargs=argparse.REMAINDER, help="-- then the module's own CLI arguments")
    a = ap.parse_args()
    rest = a.rest[1:] if a.rest[:1] == ["--"] else a.rest
    # The module's parser takes the LAST --survivors in any spelling (--survivors=PATH, or an
    # abbreviation such as --surv). The rewrite below and the driver's L3 check key on the literal
    # token, so any other spelling would let the module read a file nobody checked (Beta
    # X-E2E-WF-VOTE R1 Q2; lead ADDENDUM_4 Fix 3). Refused before the module runs.
    odd = [t for t in rest if t.startswith("--surv") and t != "--survivors"
           and ("--survivors".startswith(t.split("=", 1)[0]))]
    if rest.count("--survivors") > 1:
        odd.append("--survivors (given %d times)" % rest.count("--survivors"))
    if odd and os.environ.get("E2E_SHIM_MUTANT") != "survivors_eq_allowed":
        raise SystemExit(f"[E2E-SHIM] refuse: {odd} names the survivors file in a form the module "
                         f"would take but the shim and the driver do not check; use one literal "
                         f"'--survivors PATH'")
    cap = os.path.abspath(a.capture_dir)
    os.makedirs(cap, exist_ok=False)  # fail closed on reuse
    script = os.path.abspath(a.script)

    target = {"field": a.target_field, "default": DEFAULT_TARGET,
              "module_key": MODULE_TARGET_KEY, "rewritten": False}
    if a.target_field != DEFAULT_TARGET:
        if rest.count("--survivors") != 1 or rest.index("--survivors") + 1 >= len(rest):
            raise SystemExit("[E2E-SHIM] refuse: a non-default --target-field needs exactly one "
                             "'--survivors PATH' in the module arguments")
        j = rest.index("--survivors") + 1
        src = os.path.abspath(rest[j])
        tag = a.target_field.replace(".", "_")
        dst = os.path.join(cap, f"survivors_target_{tag}.json")
        target.update(derive_target_file(src, dst, a.target_field))
        target.update(rewritten=True, original_survivors=src, original_sha256=_sha_file(src),
                      derived_survivors=dst, derived_sha256=_sha_file(dst))
        if os.environ.get("E2E_SHIM_MUTANT") == "target_not_passed":   # self-test mutant only
            target["mutant"] = "target_not_passed"
        else:
            rest = rest[:j] + [dst] + rest[j + 1:]
        print(f"[E2E-SHIM] target {a.target_field}: derived survivors {dst} "
              f"(sha256 {target['derived_sha256']}, {target['rows']} rows)", flush=True)

    calls = []
    first_perm = {}

    # --- 1. seed and record the first permutation -------------------------
    np.random.seed(a.split_seed)
    _orig_perm = np.random.permutation

    def _perm(x):
        out = _orig_perm(x)
        if not first_perm:
            arr = np.asarray(out)
            first_perm.update({
                "n": int(arr.size),
                "sha256": hashlib.sha256(np.ascontiguousarray(arr.astype(np.int64)).tobytes()).hexdigest(),
                "head": [int(v) for v in arr[:5]],
            })
        else:
            first_perm.setdefault("later_calls", 0)
            first_perm["later_calls"] += 1
        return out
    np.random.permutation = _perm

    # --- 2. record child processes -----------------------------------------
    _orig_run = subprocess.run

    def _run(cmd, *args, **kw):
        t0 = time.time()
        proc = _orig_run(cmd, *args, **kw)
        i = len(calls)
        cwd = kw.get("cwd") or os.getcwd()
        argv = list(cmd) if isinstance(cmd, (list, tuple)) else [str(cmd)]
        script_arg = argv[1] if len(argv) > 1 else None
        resolved = os.path.realpath(os.path.join(cwd, script_arg)) if script_arg else None
        rec = {
            "i": i, "argv": argv, "cwd": os.path.abspath(cwd), "script_resolved": resolved,
            "script_sha256": (hashlib.sha256(open(resolved, "rb").read()).hexdigest()
                              if resolved and os.path.isfile(resolved) else None),
            "returncode": proc.returncode, "seconds": round(time.time() - t0, 3),
            "env_cuda_visible_devices": (kw.get("env") or os.environ).get("CUDA_VISIBLE_DEVICES"),
        }
        for name in ("stdout", "stderr"):
            v = getattr(proc, name, None)
            if isinstance(v, bytes):
                v = v.decode("utf-8", "replace")
            if v is not None:
                p = os.path.join(cap, f"call_{i:04d}.{name}.txt")
                with open(p, "w") as f:
                    f.write(v)
                rec[name + "_file"] = os.path.basename(p)
        calls.append(rec)
        with open(os.path.join(cap, "calls.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        return proc
    subprocess.run = _run

    # --- run the module unchanged ------------------------------------------
    sys.argv = [script] + rest
    rc = 0
    t0 = time.time()
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except BaseException as e:
        rc = 1
        import traceback
        traceback.print_exc()
    finally:
        summary = {
            "kind": "e2e_step5_shim", "label": "NON-CERTIFYING DIAGNOSTIC",
            "script": script,
            "script_sha256": hashlib.sha256(open(script, "rb").read()).hexdigest(),
            "argv": rest, "cwd": os.getcwd(), "split_seed": a.split_seed, "target": target,
            "first_permutation": first_perm, "exit_code": rc,
            "wall_seconds": round(time.time() - t0, 3),
            "child_calls": len(calls),
            "child_failures": sum(1 for c in calls if c["returncode"] != 0),
            "child_scripts": sorted({c["script_resolved"] for c in calls if c["script_resolved"]}),
        }
        with open(os.path.join(cap, "shim_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[E2E-SHIM] exit {rc}; child calls {summary['child_calls']}, "
              f"failures {summary['child_failures']}; first permutation {first_perm.get('sha256')}",
              flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
