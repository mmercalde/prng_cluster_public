#!/usr/bin/env python3
"""
CH14-DIAG wave 1 mutant runner (2026-11-23). CPU only.

For each mutant:
  1. build a code root under data/ch14/mutants/<id>/: symlinks to every top-level entry of the
     worktree, except the mutated file, which is a real copy with ONE textual replacement
     (the old text must occur exactly once -> "applied");
  2. the replacement calls ch14_mutant_mark.hit(), which appends to a marker file
     ("ran the mutated path");
  3. run the named test node with CH14_CODE_ROOT=<mutant root>; the mutant is KILLED only if
     pytest fails AND the failure output contains the mutant's [CREDIT:<tag>] AND the marker
     file is non-empty ("reached the credited assertion").

Runs mutants in parallel (--jobs, default min(4, cpu_count)). Writes a JSON summary.

    python3 tests/ch14_wave1_mutants.py --out data/ch14/mutants_<ts>/   (a new directory; all output under it)
"""
import argparse
import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[1]
TESTS = "tests/test_ch14_diag_wave1.py"
HIT = "__import__('ch14_mutant_mark').hit()"

MUTANTS = [
    # id, file, old, new, test node, credited tag
    ("M01_cd01_key", "training_health_check.py",
     "    if 'best_val_round' in ts:\n",
     f"    if {HIT} and False:\n",
     "test_cd01_fixed_early_stop_key", "CD01_KEY"),
    ("M01b_cd01_empty_report_graded_ok", "training_health_check.py",
     "        if pass_settings is not None and sev == 'ok' and not (diag.get('training_summary') or {}).get('rounds_captured'):\n",
     f"        if {HIT} and False:\n",
     "test_cd01_fixed_early_stop_key", "CD01_EMPTY"),
    ("M02_cd02_reader_old_key", "training_health_check.py",
     "    spread = gh.get('gradient_spread_ratio', nn_data.get('feature_gradient_spread'))\n",
     f"    spread = ({HIT} and nn_data.get('feature_gradient_spread'))\n",
     "test_cd02_reader_spread_key", "CD02_CRIT"),
    ("M03_cd02_producer_last_only", "training_diagnostics.py",
     '            return self._last_round_with("feature_gradients")\n',
     f'            return {HIT} and (self._round_data[-1] if "feature_gradients" in self._round_data[-1] else None)\n',
     "test_cd02_producer_last_round_with_spread", "CD02_PRODUCER"),
    ("M04_cd03_probe_skipped", "training_diagnostics.py",
     "                if self._probe is not None:\n",
     f"                if {HIT} and False:\n",
     "test_cd03_nn_bit_identical_and_probe[full]", "CD03_GRADS"),
    ("M05_cd03_probe_in_train_mode", "training_diagnostics.py",
     "            model.eval()\n            x = X_probe.detach()",
     f"            {HIT}; model.train()\n            x = X_probe.detach()",
     "test_cd03_nn_bit_identical_and_probe[full]", "CD03_BITID"),
    ("M06_cd04_worker_drops_names", "train_single_trial.py",
     "        _npz_names = [str(n) for n in data['feature_names']] if 'feature_names' in data.files else None\n",
     f"        _npz_names = {HIT} and None\n",
     "test_cd04_worker_reads_names_from_npz[True]", "CD04_NAMES"),
    ("M07_cd04_mpo_npz_no_names", "meta_prediction_optimizer_anti_overfit.py",
     "        if self._ch14_full():\n            # [CH14-DIAG CD-04]",
     f"        if {HIT} and False:\n            # [CH14-DIAG CD-04]",
     "test_cd04_mpo_npz_handoff[full]", "CD04_NPZ"),
    ("M08_cd05_worker_full_branch_off", "train_single_trial.py",
     "    if _diag_full(diagnostics_options) and data is not None:\n",
     f"    if {HIT} and False:\n",
     "test_cd05_worker_tree_report_and_bit_identity[xgboost]", "CD05_WORKER_xgboost"),
    ("M09_cd05_xgb_curve_last_point_only", "training_diagnostics.py",
     "        total = int(booster.num_boosted_rounds())\n        pts, stride = _stage_points(total, max_points)\n",
     f"        total = int(booster.num_boosted_rounds())\n        pts, stride = ({HIT} and [total]), 1\n",
     "test_cd05_worker_tree_report_and_bit_identity[xgboost]", "CD05_WORKER_xgboost"),
    ("M10_cd05_lgb_no_record_callback", "train_single_trial.py",
     "    _lgb_extra_cb = [lgb.record_evaluation(_lgb_evals)] if (enable_diagnostics and _diag_full(diagnostics_options)) else []\n",
     f"    _lgb_extra_cb = {HIT} and []\n",
     "test_cd05_worker_tree_report_and_bit_identity[lightgbm]", "CD05_LGB_AFTER_BEST"),
    ("M11_cd05_lgb_callback_alters_training", "train_single_trial.py",
     "    _lgb_extra_cb = [lgb.record_evaluation(_lgb_evals)] if (enable_diagnostics and _diag_full(diagnostics_options)) else []\n",
     f"    _lgb_extra_cb = [lgb.record_evaluation(_lgb_evals), lgb.reset_parameter(learning_rate=lambda i: 0.05)] if ({HIT} and enable_diagnostics and _diag_full(diagnostics_options)) else []\n",
     "test_cd05_worker_tree_report_and_bit_identity[lightgbm]", "CD05_BITID_lightgbm"),
    ("M12_cd05_inline_never_emits", "meta_prediction_optimizer_anti_overfit.py",
     "        if not self._ch14_full() or self.model_type not in ('xgboost', 'lightgbm', 'catboost'):\n",
     f"        if {HIT} or not self._ch14_full() or self.model_type not in ('xgboost', 'lightgbm', 'catboost'):\n",
     "test_cd05_inline_final_tree_report[full-xgboost]", "CD05_INLINE"),
    ("M13_cd06_reader_old_filename", "training_health_check.py",
     "    ptr = os.path.join(diag_dir, COMPARE_POINTER_FILENAME)\n",
     f"    ptr = os.path.join(diag_dir, {HIT} and 'compare_models_summary.json')\n",
     "test_cd06_reader_grades_winner", "CD06_WINNER"),
    ("M14_cd06_winner_dict_ignored", "training_health_check.py",
     "    if isinstance(wbe, dict) and isinstance(wbe.get('model_type'), str):\n",
     f"    if {HIT} and False:\n",
     "test_cd06_reader_grades_winner", "CD06_WINNER"),
    ("M15_cd06_pointer_not_written", "meta_prediction_optimizer_anti_overfit.py",
     '    if enable_diagnostics and args_dict.get("diagnostics_level", "legacy") == "full":\n',
     f'    if {HIT} and False:\n',
     "test_cd06_pointer_writer[full]", "CD06_POINTER"),
    ("M16_cd06_child_flag_not_threaded", "meta_prediction_optimizer_anti_overfit.py",
     # wave 2 (2026-11-23): the flags now come from td.diagnostics_child_flags; same test, same tag
     '            cmd.extend(_td.diagnostics_child_flags(args_dict.get("diagnostics_options")))\n',
     f'            {HIT}\n',
     "test_cd06_pointer_writer[full]", "CD06_CHILD_FLAG"),
    ("M17_cd13_classifier_off", "training_diagnostics.py",
     "    constant = (col_min == col_max) | (np.isnan(col_min) & np.isnan(col_max))\n",
     f"    constant = {HIT} and np.zeros(n_cols, dtype=bool)\n",
     "test_cd13_column_classes", "CD13_CLASSES"),
    ("M18_cd00_fixed_column_uses_legacy", "training_health_check.py",
     "        g_fixed = grade_report(report, pol_fixed, 'fixed')\n",
     f"        g_fixed = {HIT} and grade_report(report, pol_fixed, 'legacy')\n",
     "test_cd00_regrade_counts_and_read_only", "CD00_COUNTS"),
    ("M19_cd00_writes_input_and_skips_guard", "training_health_check.py",
     "    after = {f: _sha256_file(f) for f in files}\n",
     f"    {HIT}; open(files[0], 'a').write(' '); after = before\n",
     "test_cd00_regrade_counts_and_read_only", "CD00_READONLY"),
    ("M20_legacy_rules_silently_fixed", "training_health_check.py",
     "    best_round = training_summary.get('best_round', 0)\n",
     f"    best_round = {HIT} and training_summary.get('best_val_round', 0)\n",
     "test_legacy_rules_identical_to_base[late_best]", "LEGACY_IDENTITY"),
    ("M21_legacy_report_gets_full_fields", "training_diagnostics.py",
     '            if self.diagnostics_level == "full":\n                # CH14-DIAG wave 1: additive',
     f'            if {HIT} or self.diagnostics_level == "full":\n                # CH14-DIAG wave 1: additive',
     "test_nn_legacy_report_identical_to_base", "LEGACY_REPORT_IDENTITY"),
    # landing re-fit 2026-11-25: one mutant for each gate that had none
    ("M22_cd06_all_families_scope_ignored", "training_health_check.py",
     "    if per_family and settings['grade_scope'] == 'all_families' and worst_family:\n",
     f"    if {HIT} and False:\n",
     "test_cd06_all_families_scope", "CD06_ALL"),
    ("M23_cd06_absent_winner_not_absent", "training_health_check.py",
     "    if not os.path.isfile(report_path):\n        note = ('Diagnostics not enabled for this run' if report_source == 'canonical'\n",
     f"    if {HIT} and not os.path.isfile(report_path) and report_source == 'canonical':\n        note = ('Diagnostics not enabled for this run' if report_source == 'canonical'\n",
     "test_cd06_winner_report_absent_is_absent_not_canonical", "CD06_ABSENT"),
    ("M24_tree_legacy_report_gets_full_fields", "training_diagnostics.py",
     '            if self.diagnostics_level == "full":\n                # CH14-DIAG wave 1: additive',
     f'            if {HIT} or self.diagnostics_level == "full":\n                # CH14-DIAG wave 1: additive',
     "test_worker_tree_legacy_report_identical_to_base[xgboost]", "LEGACY_REPORT_IDENTITY"),
]

MARK_MODULE = '''import os
def hit():
    with open(os.environ["CH14_MUTANT_MARK"], "a") as f:
        f.write("hit\\n")
    return True
'''


def build_root(mid, fname, old, new, base):
    root = base / mid
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    for entry in WORKTREE.iterdir():
        if entry.name in (fname, ".git", "data"):
            continue
        (root / entry.name).symlink_to(entry)
    src = (WORKTREE / fname).read_text()
    n = src.count(old)
    if n != 1:
        raise RuntimeError(f"{mid}: old text occurs {n} times in {fname} (must be 1)")
    (root / fname).write_text(src.replace(old, new))
    (root / "ch14_mutant_mark.py").write_text(MARK_MODULE)
    return root


def run_one(m, base, basetemp):
    mid, fname, old, new, node, tag = m
    t0 = time.time()
    root = build_root(mid, fname, old, new, base)
    mark = base / f"{mid}.mark"
    if mark.exists():
        mark.unlink()
    env = dict(os.environ, CH14_CODE_ROOT=str(root), CH14_MUTANT_MARK=str(mark),
               CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    cmd = [sys.executable, "-m", "pytest", "-q", f"{TESTS}::{node}", "-p", "no:cacheprovider",
           f"--basetemp={basetemp / mid}", "-x"]
    p = subprocess.run(cmd, cwd=WORKTREE, env=env, capture_output=True, text=True, timeout=900)
    out = p.stdout + p.stderr
    hits = mark.read_text().count("hit") if mark.exists() else 0
    credited = f"[CREDIT:{tag}]" in out
    killed = p.returncode != 0 and credited and hits > 0
    (base / f"{mid}.log").write_text(out)
    return {"id": mid, "file": fname, "test": node, "credited_tag": tag, "applied": True,
            "pytest_rc": p.returncode, "marker_hits": hits, "credited_tag_in_failure": credited,
            "killed": killed, "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if not str(out).startswith(str(WORKTREE / "data") + "/") or out.exists():
        raise SystemExit("--out must be a NEW directory under <worktree>/data/ (everything is written there)")
    base = out / "mutants"
    basetemp = out / "mutant_pytest_tmp"
    base.mkdir(parents=True, exist_ok=True)
    basetemp.mkdir(parents=True, exist_ok=True)
    todo = [m for m in MUTANTS if not a.only or m[0] in a.only]
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        results = list(ex.map(lambda m: run_one(m, base, basetemp), todo))
    summary = {"schema": "ch14_wave1_mutants/1.0", "jobs": a.jobs, "wall_seconds": round(time.time() - t0, 1),
               "total": len(results), "killed": sum(r["killed"] for r in results), "results": results}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    for r in results:
        print(f"{'KILLED ' if r['killed'] else 'SURVIVED'} {r['id']:42s} rc={r['pytest_rc']} hits={r['marker_hits']} "
              f"tag={r['credited_tag_in_failure']} {r['seconds']}s")
    print(f"{summary['killed']}/{summary['total']} killed in {summary['wall_seconds']}s (jobs={a.jobs})")
    return 0 if summary["killed"] == summary["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
