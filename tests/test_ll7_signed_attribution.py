#!/usr/bin/env python3
"""
tests/test_ll7_signed_attribution.py - self-test for LL-7 (plan PLAN_LAND_LL7_LL8_20261126 §3):
  - signed attribution by name (per_survivor_attribution.py, taken from ch14-learn 5353a658);
  - Beta X-LL6-BUNDLE R1 Q3: under --c13-model-loader canonical, the regime-shift tier comparison
    is attributed on the bundle's model input (compare_pool_tiers(..., model_input=...), passed by
    chapter_13_orchestrator._run_regime_shift_analysis);
  - Beta X-LL6-BUNDLE R1 Finding 2(a): under canonical, a positional feature list is refused, even
    at the sidecar's width (chapter_13_orchestrator._c13_model_input).

Standard (LL-6): every "unchanged" gate compares with a baseline read by `git show <full sha>:<file>`
(--baseline-rev, default 087bc36..., main before LL-7), never HEAD; TMPDIR, XDG_CACHE_HOME,
MPLCONFIGDIR and CUDA_CACHE_PATH under --out before any library import; a no-write gate on the exact
boundary with a sibling-write negative control; source mutants (exact text replacements in a copy, code
dir first on sys.path), each credited only if its target gates are green unmutated and all turn red.

Fixtures: tests/test_ll6_inference_bundle.build_fixtures (synthetic rows, Step 3 worker, four models
trained by the real Step 5 CLI on CPU), imported, not copied; that file is not edited.

CPU only. Writes only under --out, which must lie under <this tree>/data/.

  PYTHONDONTWRITEBYTECODE=1 python3 -u -B tests/test_ll7_signed_attribution.py --out data/ll78_build/ll7_<stamp>
"""
import argparse
import hashlib
import json
import logging
import os
import random
import re
import resource
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tests"))
import test_ll6_inference_bundle as ll6  # noqa: E402  (stdlib-only at import; helpers reused, not copied)

FAMILIES = ll6.FAMILIES
BASELINE_REV = "087bc366fb199dbbe26c1fe918a911158618d358"   # main before LL-7 (full sha)
A_NORM_ATOL = 1e-5            # target units, as LL-6
COMPLETENESS_ATOL = 1e-4      # = per_survivor_attribution.COMPLETENESS_ATOL_TREE (declared before measurement)
BATCH_ROW_ATOL = 1e-6         # ch14-learn run 1 failed at 1e-9 (3.2e-9, NN float32 summation order); re-declared
RESULTS = []
RUN_INFO = {}


def gate(name, ok, detail=""):
    RESULTS.append({"gate": name, "ok": bool(ok), "detail": str(detail)[:600]})
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {str(detail)[:400]}", flush=True)
    return ok


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# source mutants: one or more exact replacements in a copy of ONE file
# ---------------------------------------------------------------------------
MARK = '__import__("pathlib").Path(__import__("os").environ["LL7_MUTANT_MARK"]).touch()'
PSA, C13 = "per_survivor_attribution.py", "chapter_13_orchestrator.py"
# (id, file, [(old, new, count)], marker_before_first_old_line, target gates)
MUTANTS = [
    ("drop_sign", PSA,
     [('signed = {n: float(contrib[i]) for i, n in enumerate(names)}',
       'signed = {n: float(abs(contrib[i])) for i, n in enumerate(names)}', 1)], True,
     ["LL7.row_completeness_from_signed_values"]),
    ("positional_zip", PSA,
     [('z = bundle.transform(bundle.vector(features, missing=missing).reshape(1, -1))[0]',
       'z = bundle.transform(np.array(list(features.values())[:bundle.feature_count], dtype=np.float32)'
       '.reshape(1, -1))[0]', 1)], True,
     ["LL7.attribute_row_by_name_invariant_to_key_order"]),
    ("name_check_skipped", PSA,
     [('        elif mn != names:\n', '        elif False:\n', 1)], False,
     ["LL7.name_and_count_checks_fail_closed"]),
    ("batch_ig_target_units_dropped", PSA,     # batch NN path forgets the y_std rescale
     [('S[i:i + step] = (zt * avg).detach().cpu().numpy() * ys',
       'S[i:i + step] = (zt * avg).detach().cpu().numpy()', 1)], True,
     ["LL7.batch_equals_row_and_shares_sum_to_1"]),
    ("absolute_output_changed", PSA,
     [('    return backend(model, features, feature_names)',
       '    return backend(model, features[::-1].copy(), feature_names)', 1)], True,
     ["LL7.old_absolute_output_unchanged_vs_base"]),
    ("legacy_tier_changed", PSA,
     [('                features_arr = np.array(s["features"], dtype=np.float32)',
       '                features_arr = np.array(s["features"], dtype=np.float32)[::-1].copy()', 1)], True,
     ["LL7.q3_legacy_tier_and_root_cause_identical_to_base"]),
    ("tier_raw_features", PSA,
     [('                features_arr = model_input(s["features"])',
       '                features_arr = np.array(s["features"], dtype=np.float32)', 1)], True,
     ["LL7.q3_canonical_tiers_use_bundle_input"]),
    ("tier_unscaled", C13,
     [('{"model_input": (lambda f: _c13_model_input(bundle, f))}',
       '{"model_input": (lambda f: bundle.vector(f))}', 1)], False,
     ["LL7.q3_canonical_tiers_use_bundle_input"]),
    ("handoff_dropped", C13,                     # Beta X-LL6 R1 Q5 handoff mutant, for the tier handoff
     [('bundle=bundle,  # [LL-7] Q3: the tiers use the same input as above',
       'bundle=None,  # [LL-7] Q3: the tiers use the same input as above', 1)], False,
     ["LL7.q3_canonical_tiers_use_bundle_input"]),
    ("positional_accepted_at_width", C13,        # restores the LL-6 width-only acceptance
     [('    if not isinstance(features, dict):\n        raise ValueError(f"canonical root cause',
       '    if not isinstance(features, dict) and len(features) != bundle.feature_count:\n'
       '        raise ValueError(f"canonical root cause', 1),
      ('    x = bundle.vector(features)\n    return bundle.transform(x.reshape(1, -1))[0]',
       '    x = bundle.vector(features) if isinstance(features, dict) else np.asarray(features, dtype=np.float32)\n'
       '    return bundle.transform(x.reshape(1, -1))[0]', 1)], False,
     ["LL7.finding2a_positional_refused_under_canonical"]),
]


def make_mutant_dir(root, rel, reps, marker):
    src = (REPO / rel).read_text()
    for old, new, count in reps:
        n = src.count(old)
        if n != count:
            return None, f"anchor {old[:50]!r} found {n} times, want {count}"
    if marker:
        old0 = reps[0][0]
        lines, done = [], False
        for line in src.split("\n"):
            if old0 in line and not done:
                lines.append(line[:len(line) - len(line.lstrip())] + MARK)
                done = True
            lines.append(line)
        src = "\n".join(lines)
    for old, new, count in reps:
        src = src.replace(old, new)
    code = root / "code"
    code.mkdir(parents=True)
    (code / rel).write_text(src)
    return code, f"sha {sha(REPO / rel)[:12]} -> {sha(code / rel)[:12]}"


def run_mutants(out, jobs, models_src, green, baseline_rev, boundary):
    def one(m):
        mid, rel, reps, marker, targets = m
        root = out / "mutants" / mid
        unmut = {t: green.get(t) for t in targets}
        if not all(v is True for v in unmut.values()):
            return {"id": mid, "killed": False, "targets": targets, "unmutated": unmut,
                    "detail": "a target gate is not green in the unmutated run: no credit"}
        code, how = make_mutant_dir(root, rel, reps, marker)
        if code is None:
            return {"id": mid, "killed": False, "detail": how}
        mark = root / "mark"
        p = subprocess.run([sys.executable, "-u", "-B", str(Path(__file__).resolve()), "--out", str(root / "run"),
                            "--code-dir", str(code), "--models-from", str(models_src), "--no-mutants",
                            "--only", ",".join(targets), "--baseline-rev", baseline_rev, "--boundary", str(boundary)],
                           capture_output=True, text=True,
                           env=dict(os.environ, LL7_MUTANT_MARK=str(mark), CUDA_VISIBLE_DEVICES="",
                                    PYTHONDONTWRITEBYTECODE="1"))
        (root / "run.log").write_text(p.stdout + p.stderr)
        try:
            g = {r["gate"]: r["ok"] for r in json.load(open(root / "run" / "selftest_summary.json"))["gates"]}
        except Exception as e:
            return {"id": mid, "killed": False, "detail": f"no summary ({e}); rc {p.returncode}"}
        red = [t for t in targets if g.get(t) is False]
        killed = len(red) == len(targets) and (mark.exists() or not marker)
        return {"id": mid, "file": rel, "killed": killed, "targets_red": red, "targets": targets, "unmutated": unmut,
                "marker": ("hit" if mark.exists() else "absent") if marker else "n/a", "applied": how}
    res = []
    with ThreadPoolExecutor(jobs) as ex:
        for r in ex.map(one, MUTANTS):
            res.append(r)
            print(f"[{'PASS' if r['killed'] else 'FAIL'}] MUTANT LL7.{r['id']} killed={r['killed']} "
                  f"red={r.get('targets_red')} marker={r.get('marker')} {r.get('applied', r.get('detail'))}", flush=True)
    return res


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline-rev", default=BASELINE_REV)
    ap.add_argument("--code-dir", default=None, help="mutant runs only")
    ap.add_argument("--models-from", default=None, help="reuse fixtures from an earlier --out")
    ap.add_argument("--only", default=None)
    ap.add_argument("--no-mutants", action="store_true")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--boundary", default=None)
    ap.add_argument("--clean-on-pass", action="store_true",
                    help="after an all-green run, delete fixtures and mutant dirs; keep summary and logs")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if not str(out).startswith(str(REPO / "data") + "/"):
        raise SystemExit("--out must lie under <this tree>/data/")
    if not re.fullmatch(r"[0-9a-f]{40}", a.baseline_rev):
        raise SystemExit("--baseline-rev must be a full 40-hex sha (never HEAD, a branch or a short sha)")
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
    code_dir = Path(a.code_dir).resolve() if a.code_dir else None
    sys.path.insert(0, str(REPO))
    if code_dir:
        sys.path.insert(0, str(code_dir))
    t0 = time.time()
    guard_before = ll6.boundary_snapshot(boundary)

    # ---- baseline precondition ------------------------------------------------
    base = out / "base"
    base.mkdir()
    under = lambda rel: (code_dir / rel if code_dir and (code_dir / rel).exists() else REPO / rel)
    psa_b, c13_b = ll6.git_show(a.baseline_rev, PSA), ll6.git_show(a.baseline_rev, C13)
    psa_now, c13_now = under(PSA).read_text(), under(C13).read_text()
    pre_ok = (psa_b is not None and c13_b is not None
              and b"per_survivor_attribution_signed" not in psa_b and b"model_input=None" not in psa_b
              and "per_survivor_attribution_signed" in psa_now and "model_input" in psa_now
              and b"positional input refused" not in c13_b)
    (base / PSA).write_bytes(psa_b or b"")
    (base / C13).write_bytes(c13_b or b"")
    gate("LL7.baseline_precondition", pre_ok,
         f"baseline {a.baseline_rev[:12]} lacks the signed functions and model_input; this tree has them; "
         f"base sha {sha(base / PSA)[:12]}, {sha(base / C13)[:12]}")
    if not pre_ok:
        return finish(out, t0, [], guard_before, boundary, a)

    # ---- fixtures ---------------------------------------------------------------
    t_fix = time.time()
    if a.models_from:
        src = Path(a.models_from).resolve()
        surv, hist = src / "survivors.json", src / "fx" / "train.json"
        dirs = {f: src / f"models_{f}" / "models/reinforcement" for f in FAMILIES}
    else:
        surv, hist, dirs = ll6.build_fixtures(out)
    t_fix = time.time() - t_fix
    rows = json.load(open(surv))
    feats = [r["features"] for r in rows]

    import per_survivor_attribution as psa
    import models.inference_bundle as ib
    import chapter_13_orchestrator as c13
    psa_base = ll6.load_as("ll7_psa_base", base / PSA)
    c13_base = ll6.load_as("ll7_c13_base", base / C13)
    # the base orchestrator imported compare_pool_tiers / per_survivor_attribution from whichever module is
    # in sys.modules (this tree's); bind it to the BASE module so the baseline is the base end to end
    c13_base.compare_pool_tiers = psa_base.compare_pool_tiers
    c13_base.per_survivor_attribution = psa_base.per_survivor_attribution
    RUN_INFO["modules"] = {"psa": psa.__file__, "c13": c13.__file__}
    logging.getLogger().setLevel(logging.ERROR)
    bundles = {f: ib.load_inference_bundle(str(dirs[f])) for f in FAMILIES}

    # ================= signed attribution ========================================
    if want("LL7.tree_completeness_and_prediction_is_the_models"):
        comp, pdiff, signs = {}, {}, {}
        for fam in FAMILIES:
            b = bundles[fam]
            Z = b.transform(b.matrix(feats))
            M = psa.per_survivor_attribution_signed_matrix(b.attribution_model, fam, Z, b.feature_names,
                                                           expected_feature_count=b.feature_count,
                                                           y_mean=b.y_mean, y_std=b.y_std)
            comp[fam] = float(np.max(np.abs(M["prediction"] - (M["baseline"] + M["signed"].sum(axis=1)))))
            pdiff[fam] = float(np.max(np.abs(M["prediction"] - ll6.independent_forward(b, Z))))
            signs[fam] = (int((M["signed"] > 0).sum()), int((M["signed"] < 0).sum()))
        gate("LL7.tree_completeness_and_prediction_is_the_models",
             all(comp[f] <= COMPLETENESS_ATOL for f in ("xgboost", "lightgbm", "catboost"))
             and all(v <= A_NORM_ATOL for v in pdiff.values())
             and all(p > 0 and n > 0 for p, n in signs.values()),
             f"completeness (recomputed from S) {({f: f'{v:.1e}' for f, v in comp.items()})}; prediction vs independent "
             f"forward {({f: f'{v:.1e}' for f, v in pdiff.items()})}; (+,-) counts {signs}")
    if want("LL7.row_completeness_from_signed_values"):
        worst = {}
        for fam in ("xgboost", "lightgbm", "catboost"):
            b = bundles[fam]
            w = 0.0
            for i in range(12):
                r = psa.attribute_row(b, feats[i])
                w = max(w, abs(r["prediction"] - (r["baseline"] + sum(r["signed"].values()))))
            worst[fam] = w
        gate("LL7.row_completeness_from_signed_values", all(v <= COMPLETENESS_ATOL for v in worst.values()),
             f"max |pred - (baseline + sum(signed))| per family, recomputed from the returned signed map: "
             f"{({f: f'{v:.1e}' for f, v in worst.items()})}")
    if want("LL7.nn_integrated_gradients_residual_shrinks_with_steps"):
        b = bundles["neural_net"]
        Zs = b.transform(b.matrix(feats[:40]))
        e = {}
        for st in (8, 64, 512):
            r = psa.per_survivor_attribution_signed_matrix(b.attribution_model, "neural_net", Zs, b.feature_names,
                                                           y_mean=b.y_mean, y_std=b.y_std, ig_steps=st)
            e[st] = float(np.abs(r["completeness_residual"]).max())
        gate("LL7.nn_integrated_gradients_residual_shrinks_with_steps", e[512] < e[64] < e[8],
             f"max |residual| 8 steps {e[8]:.2e}, 64 {e[64]:.2e}, 512 {e[512]:.2e} (reported, not exact); y_std {b.y_std}")
    if want("LL7.batch_equals_row_and_shares_sum_to_1"):
        worst, sh = 0.0, 0.0
        for fam in FAMILIES:
            b = bundles[fam]
            M = psa.per_survivor_attribution_signed_matrix(b.attribution_model, fam, b.transform(b.matrix(feats[:12])),
                                                           b.feature_names, y_mean=b.y_mean, y_std=b.y_std)
            for i in range(12):
                r = psa.attribute_row(b, feats[i])
                worst = max(worst, max(abs(M["signed"][i][j] - r["signed"][n]) for j, n in enumerate(b.feature_names)),
                            abs(M["prediction"][i] - r["prediction"]))
                sh = max(sh, abs(sum(r["abs_share"].values()) - 1))
        gate("LL7.batch_equals_row_and_shares_sum_to_1", worst <= BATCH_ROW_ATOL and sh <= 1e-9,
             f"max |batch - row| {worst:.2e} (atol {BATCH_ROW_ATOL}); max |sum(shares) - 1| {sh:.1e}")
    if want("LL7.old_absolute_output_unchanged_vs_base"):
        same, nonempty = True, True
        for fam in FAMILIES:
            b = bundles[fam]
            for i in range(5):
                x = b.transform(b.matrix(feats[i:i + 1]))[0]
                new = psa.per_survivor_attribution(b.attribution_model, fam, x, b.feature_names)
                old = psa_base.per_survivor_attribution(b.attribution_model, fam, x, b.feature_names)
                same = same and new == old
                nonempty = nonempty and bool(old) and len(set(old.values())) > 1
        gate("LL7.old_absolute_output_unchanged_vs_base", same and nonempty,
             f"4 families x 5 rows, dicts identical to {a.baseline_rev[:12]}: {same}; non-empty, not constant: {nonempty}")
    if want("LL7.name_and_count_checks_fail_closed"):
        refused = []
        b = bundles["lightgbm"]
        z = b.transform(b.matrix(feats[:1]))[0]
        nm = list(b.feature_names)
        cases = (("count", lambda: psa.per_survivor_attribution_signed(b.attribution_model, "lightgbm", z[:-1], nm[:-1])),
                 ("duplicate", lambda: psa.per_survivor_attribution_signed(b.attribution_model, "lightgbm", z,
                                                                          nm[:-1] + nm[:1])),
                 ("missing name", lambda: psa.attribute_row(b, {k: v for k, v in feats[0].items() if k != nm[5]})),
                 ("list not map", lambda: psa.attribute_row(b, [feats[0][n] for n in nm])),
                 ("model names differ", lambda: psa._check_names(nm, z, len(nm), list(reversed(nm)))))
        for label, fn in cases:
            try:
                fn()
            except (psa.AttributionInputError, ib.BundleError):
                refused.append(label)
        ctrl = psa._check_names(nm, z, len(nm), nm)
        gate("LL7.name_and_count_checks_fail_closed", len(refused) == len(cases) and ctrl == "equal",
             f"refused {refused}; control (same names) -> {ctrl!r}")
    if want("LL7.attribute_row_by_name_invariant_to_key_order"):
        diffs, reordered = {}, 0
        for fam in ("neural_net", "catboost"):
            b = bundles[fam]
            d = 0.0
            for i in range(6):
                ks = list(feats[i])
                random.Random(5 + i).shuffle(ks)
                shuf = {k: feats[i][k] for k in ks}
                reordered += [shuf[n] for n in b.feature_names] != list(shuf.values())[:b.feature_count]
                r1, r2 = psa.attribute_row(b, feats[i]), psa.attribute_row(b, shuf)
                d = max(d, abs(r1["prediction"] - r2["prediction"]),
                        max(abs(r1["signed"][n] - r2["signed"][n]) for n in b.feature_names))
            diffs[fam] = d
        gate("LL7.attribute_row_by_name_invariant_to_key_order", all(v == 0.0 for v in diffs.values()) and reordered == 12,
             f"max |diff| original vs shuffled key order {diffs}; non-vacuity: value order differs from sidecar "
             f"order in {reordered}/12 shuffled rows")

    # ================= Chapter 13: Q3 and Finding 2(a) ===========================
    def root_cause(mod, preds, model, fam, names, bundle, tag, force_regime=True):
        d = out / "c13_cwd" / tag
        d.mkdir(parents=True)
        cwd = os.getcwd()
        os.chdir(d)
        orig = mod.per_survivor_attribution
        cnt = {"n": 0}

        def forced(model, model_type, features, feature_names):   # steps 1-2 only: misses lean on A, hits on B
            orig(model, model_type, features, feature_names)       # the real call still runs (input checked)
            cnt["n"] += 1
            k = 0 if cnt["n"] <= 18 else 3
            return {feature_names[k + j]: 1.0 / (j + 1) for j in range(3)}
        if force_regime:
            mod.per_survivor_attribution = forced
        try:
            o = object.__new__(mod.Chapter13Orchestrator)
            kw = {"bundle": bundle} if bundle is not None else {}
            r = mod.Chapter13Orchestrator.post_draw_root_cause_analysis(o, {"draw_id": "ll7_fixture"}, preds, model,
                                                                         fam, names, **kw)
            tp = d / "diagnostics_outputs" / "tier_comparison.json"
            return r, (json.load(open(tp)) if tp.exists() else None)
        finally:
            mod.per_survivor_attribution = orig
            os.chdir(cwd)

    strip_meta = lambda t: {k: v for k, v in (t or {}).items() if k != "metadata"}
    strip_ts = lambda r: {k: v for k, v in (r or {}).items() if k not in ("timestamp", "tier_comparison_path")}

    def tier_dist(t1, t2):
        if not t1 or not t2:
            return float("inf")
        return max(abs(t1[k][f] - t2[k][f]) for k in t1 if k != "metadata" for f in t1[k])

    if want("LL7.q3_canonical_tiers_use_bundle_input"):
        det = {}
        ok = True
        for fam in ("neural_net", "lightgbm"):
            b = bundles[fam]
            scores = b.predict(b.matrix(feats))
            preds = [{"seed": r["seed"], "rank": i + 1, "hit": i in (2, 7), "prediction": float(scores[i]),
                      "features": dict(r["features"])} for i, r in enumerate(rows)]
            res, tier = root_cause(c13, preds, b.attribution_model, fam, b.feature_names, b, f"q3_canonical_{fam}")
            Zp = [{"prediction": p["prediction"], "features": z} for p, z in
                  zip(preds, b.transform(b.matrix([p["features"] for p in preds])))]
            want_t = psa.compare_pool_tiers(b.attribution_model, fam, Zp, b.feature_names)
            raw = [{"prediction": p["prediction"], "features": b.vector(p["features"])} for p in preds]
            unscaled = psa.compare_pool_tiers(b.attribution_model, fam, raw, b.feature_names)
            eq = tier is not None and strip_meta(tier) == strip_meta(want_t)
            sep = tier_dist(want_t, unscaled)
            fails = (tier or {}).get("metadata", {}).get("attribution_failures")
            ok = ok and eq and res is not None and res["diagnosis"] == "regime_shift" and fails == 0 \
                and (sep > 1e-6 if fam == "neural_net" else True)
            det[fam] = {"tier_written": tier is not None, "equal_to_bundle_input_tiers": eq,
                        "diagnosis": res and res["diagnosis"], "attribution_failures": fails,
                        "bundle_vs_unscaled_tier_max_diff": f"{sep:.2e}"}
        gate("LL7.q3_canonical_tiers_use_bundle_input", ok,
             f"through post_draw_root_cause_analysis (canonical, name-keyed rows, forced regime shift): {det}; "
             f"non-vacuity: NN tiers on unscaled vectors differ")
    if want("LL7.q3_legacy_tier_and_root_cause_identical_to_base"):
        det = {}
        ok = True
        for fam in ("neural_net", "lightgbm"):
            b = bundles[fam]
            scores = b.predict(b.matrix(feats))
            preds = [{"seed": r["seed"], "rank": i + 1, "hit": i in (2, 7), "prediction": float(scores[i]),
                      "features": [float(r["features"].get(n, 0.0)) for n in b.feature_names]}
                     for i, r in enumerate(rows)]
            r_new, t_new = root_cause(c13, preds, b.attribution_model, fam, b.feature_names, None, f"leg_new_{fam}")
            r_old, t_old = root_cause(c13_base, preds, b.attribution_model, fam, b.feature_names, None, f"leg_base_{fam}")
            direct_new = psa.compare_pool_tiers(b.attribution_model, fam, preds, b.feature_names)
            direct_old = psa_base.compare_pool_tiers(b.attribution_model, fam, preds, b.feature_names)
            nz = t_old is not None and any(v != 0 for f, v in t_old["top_20"].items())
            e = (t_old is not None and strip_meta(t_new) == strip_meta(t_old) and strip_ts(r_new) == strip_ts(r_old)
                 and strip_meta(direct_new) == strip_meta(direct_old) and nz)
            ok = ok and e
            det[fam] = {"equal": e, "diagnosis": r_old and r_old["diagnosis"], "tier_nonzero": nz}
        gate("LL7.q3_legacy_tier_and_root_cause_identical_to_base", ok,
             f"legacy (no bundle, positional rows): tier file, root-cause result and a direct compare_pool_tiers "
             f"call equal to {a.baseline_rev[:12]} (timestamps aside): {det}")
    if want("LL7.finding2a_positional_refused_under_canonical"):
        b = bundles["neural_net"]
        f0 = feats[0]
        same_order = [float(f0[n]) for n in b.feature_names]
        rev = list(reversed(same_order))
        out_r = {}
        for label, x in (("sidecar order", same_order), ("reordered", rev), ("ndarray", np.array(same_order))):
            try:
                c13._c13_model_input(b, x)
                out_r[label] = "accepted"
            except ValueError:
                out_r[label] = "refused"
        dict_ok = np.array_equal(c13._c13_model_input(b, dict(f0)), b.transform(b.matrix([f0]))[0])
        try:
            base_acc = c13_base._c13_model_input(b, rev) is not None
        except ValueError:
            base_acc = False
        preds_pos = [{"seed": r["seed"], "rank": i + 1, "hit": i in (2, 7),
                      "features": [float(r["features"][n]) for n in b.feature_names]} for i, r in enumerate(rows[:20])]
        r_pos, _ = root_cause(c13, preds_pos, b.attribution_model, "neural_net", b.feature_names, b, "f2a_pos",
                              force_regime=False)
        preds_dict = [dict(p, features=dict(rows[i]["features"])) for i, p in enumerate(preds_pos)]
        r_dict, _ = root_cause(c13, preds_dict, b.attribution_model, "neural_net", b.feature_names, b, "f2a_dict",
                               force_regime=False)
        gate("LL7.finding2a_positional_refused_under_canonical",
             set(out_r.values()) == {"refused"} and dict_ok and base_acc and r_pos is None and r_dict is not None,
             f"_c13_model_input: {out_r}; name-keyed map accepted and equal to the bundle input: {dict_ok}; "
             f"non-vacuity: the base ({a.baseline_rev[:12]}) accepts the same-width reordered list: {base_acc}; "
             f"root cause on positional rows -> {r_pos} (refused, logged), on name-keyed rows -> "
             f"{r_dict and r_dict['attribution_success']}")
    if want("LL7.c13_default_still_legacy"):
        import inspect
        d = inspect.signature(c13.Chapter13Orchestrator.__init__).parameters["c13_model_loader"].default
        cli = re.search(r'add_argument\("--c13-model-loader".*?default="(\w+)"', c13_now, re.S)
        gate("LL7.c13_default_still_legacy", d == "legacy" and cli is not None and cli.group(1) == "legacy",
             f"constructor default {d!r}; CLI default {cli and cli.group(1)!r}")

    # ================= Step 6: pool output ========================================
    if want("LL7.step6_does_not_import_changed_files"):
        probe = ("import sys, json; sys.path.insert(0, sys.argv[1]); import prediction_generator; "
                 "print(json.dumps(sorted(m for m in ('per_survivor_attribution', 'chapter_13_orchestrator') "
                 "if m in sys.modules)))")
        p = subprocess.run([sys.executable, "-B", "-c", probe, str(REPO)], capture_output=True, text=True,
                           cwd=str(out), env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"))
        try:
            got = json.loads(p.stdout.strip().splitlines()[-1])
        except Exception:
            got = None
        src = (REPO / "prediction_generator.py").read_text()
        gate("LL7.step6_does_not_import_changed_files",
             got == [] and "per_survivor_attribution" not in src and "chapter_13_orchestrator" not in src,
             f"modules loaded by importing prediction_generator: {got}; names absent from its source: "
             f"{'per_survivor_attribution' not in src and 'chapter_13_orchestrator' not in src}")
    if want("LL7.step6_cli_pool_output_identical_to_base"):
        bdir = out / "base_code"
        bdir.mkdir()
        shutil.copy(base / PSA, bdir / PSA)
        shutil.copy(base / C13, bdir / C13)
        pgf = REPO / "prediction_generator.py"
        det = {}
        ok = True
        for fam in ("neural_net", "xgboost"):
            r_b1 = ll6.step6_cli(out, f"{fam}_base1", pgf, bdir, dirs[fam], surv, hist)
            r_b2 = ll6.step6_cli(out, f"{fam}_base2", pgf, bdir, dirs[fam], surv, hist)
            r_n = ll6.step6_cli(out, f"{fam}_new", pgf, code_dir, dirs[fam], surv, hist)
            fb1, fb2, fn = (ll6.flatten(r.get("result")) for r in (r_b1, r_b2, r_n))
            vol = {k for k in set(fb1) | set(fb2) if fb1.get(k) != fb2.get(k)}
            bad_vol = [k for k in vol if not re.search(r"time|stamp|_at$|date|run_id", k, re.I)]
            diff = [k for k in set(fb1) | set(fn) if k not in vol and fb1.get(k) != fn.get(k)]
            pool = (r_n.get("result") or {}).get("predictions") or (r_n.get("result") or {}).get("pool")
            base_ok = all(r.get("rc") == 0 and r.get("result") is not None for r in (r_b1, r_b2))  # Beta X-LL7-LL8 R1 Q6
            e = (base_ok and r_n.get("rc") == 0 and r_n.get("result") is not None and not diff
                 and not bad_vol)
            ok = ok and e
            det[fam] = {"equal": e, "both_base_runs_ok": base_ok, "keys": len(fn), "volatile": sorted(vol)[:4], "differs": diff[:4],
                        "pool_entries": len(pool) if isinstance(pool, list) else None}
        gate("LL7.step6_cli_pool_output_identical_to_base", ok,
             f"Step 6 CLI (tests/ll6_step6_runner.py) with base per_survivor_attribution + orchestrator vs this "
             f"tree: {det}")

    mutants = []
    if not a.no_mutants and code_dir is None:
        green = {r["gate"]: r["ok"] for r in RESULTS}
        mutants = run_mutants(out, a.jobs, Path(a.models_from).resolve() if a.models_from else out, green,
                              a.baseline_rev, boundary)
    if code_dir is None and want("LL7.no_write_gate_catches_sibling_write"):
        s0 = ll6.boundary_snapshot(boundary)
        sent = REPO / "data" / f"ll7_nowrite_control_{os.getpid()}_{time.time_ns()}"
        sent.mkdir()
        (sent / "sentinel.txt").write_text("ll7 no-write negative control\n")
        s1 = ll6.boundary_snapshot(boundary)
        shutil.rmtree(sent)
        s2 = ll6.boundary_snapshot(boundary)
        caught, cleared = ll6.boundary_diff(s0, s1), ll6.boundary_diff(s0, s2)
        gate("LL7.no_write_gate_catches_sibling_write",
             any(str(sent.relative_to(REPO)) in c for c in caught) and not cleared,
             f"sibling sentinel seen: {caught[:2]}; after removal clean: {not cleared}")
    RUN_INFO["fixture_seconds"] = round(t_fix, 1)
    return finish(out, t0, mutants, guard_before, boundary, a)


def finish(out, t0, mutants, guard_before, boundary, a):
    ch = ll6.boundary_diff(guard_before, ll6.boundary_snapshot(boundary))
    gate("LL7.no_write_outside_out", not ch, f"changes outside {boundary}: {ch[:5]}")
    n_ok = sum(r["ok"] for r in RESULTS)
    k_ok = sum(m["killed"] for m in mutants)
    summ = {"gates": RESULTS, "mutants": mutants, "gates_pass": n_ok, "gates_total": len(RESULTS),
            "mutants_killed": k_ok, "mutants_total": len(mutants), "wall_s": round(time.time() - t0, 1),
            "rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
            "rss_children_mb": round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024),
            "run_info": RUN_INFO}
    all_ok = n_ok == len(RESULTS) and k_ok == len(mutants) and (a.no_mutants or a.code_dir or a.only
                                                                 or len(mutants) == len(MUTANTS))
    (out / "selftest_summary.json").write_text(json.dumps(summ, indent=1, default=str))
    print(f"\nLL-7 self-test: {n_ok}/{len(RESULTS)} gates, {k_ok}/{len(mutants)} mutants killed, "
          f"{summ['wall_s']} s, RSS {summ['rss_mb']} MB (largest child {summ['rss_children_mb']} MB) -> "
          f"{'PASS' if all_ok else 'FAIL'}", flush=True)
    if all_ok and a.clean_on_pass and not a.code_dir:
        for d in sorted(out.iterdir()):
            if d.is_dir() and d.name != "_env":
                shutil.rmtree(d)
        shutil.rmtree(out / "_env", ignore_errors=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
