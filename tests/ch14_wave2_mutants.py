#!/usr/bin/env python3
"""
CH14-DIAG wave 2 mutant runner (2026-11-23). CPU only. Same rules as tests/ch14_wave1_mutants.py
(one exact textual replacement per mutant, a marker proves the mutated line ran, KILLED only if
pytest fails with the credited [CREDIT:<tag>] in the failure output), against
tests/test_ch14_diag_wave2.py. At most 2 jobs (lead's load note, 2026-11-23 19:10); run it under
`taskset -c 4,5 nice -n 15`.

    taskset -c 4,5 nice -n 15 python3 tests/ch14_wave2_mutants.py --out data/ch14/w2/mutants_<ts>/   (a new directory; all output under it)
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ch14_wave1_mutants as w1  # noqa: E402  (build_root / run_one machinery)

WORKTREE = w1.WORKTREE
w1.TESTS = "tests/test_ch14_diag_wave2.py"
HIT = w1.HIT

MUTANTS = [
    # id, file, old, new, test node, credited tag
    ("W01_cd07_no_freshness_check", "training_health_check.py",
     "    rid = report.get('diagnostics_run_id')\n",
     f"    return {{'freshness': 'unverified', 'reason': 'mutant', 'report_run_id': None}} if {HIT} else None\n"
     "    rid = report.get('diagnostics_run_id')\n",
     "test_cd07_fixed_path_stale_winner_not_graded", "CD07_STALE"),
    ("W02_cd07_family_not_checked", "training_health_check.py",
     "            if ffr['freshness'] == 'stale':\n",
     f"            if {HIT} and False:\n",
     "test_cd07_stale_family_listed_not_graded", "CD07_FAMILY"),
    ("W03_cd07_stale_counts_toward_skip", "training_health_check.py",
     "    action = _wave2_settings(policies)['stale_report_action']\n    if action == 'grade':\n",
     f"    action = _wave2_settings(policies)['stale_report_action']\n    if {HIT} or action == 'grade':\n",
     "test_cd07_fixed_path_stale_winner_not_graded", "CD07_STALE"),
    ("W04_cd07_pointer_unstamped", "meta_prediction_optimizer_anti_overfit.py",
     '            json.dump({**summary, "diagnostics_run_id": _dopts.get("run_id"),\n',
     f'            json.dump({{**summary, "diagnostics_run_id": {HIT} and None,\n',
     "test_cd07_pointer_and_children_share_run_id", "CD07_POINTER"),
    ("W05_cd07_children_get_new_ids", "training_diagnostics.py",
     '             ("run_id", "--diagnostics-run-id", str),\n',
     f'             ("run_id", "--diagnostics-run-id", lambda v: {HIT} and new_run_id()[0]),\n',
     "test_cd07_pointer_and_children_share_run_id", "CD07_POINTER"),
    ("W06_cd07_report_unstamped", "training_diagnostics.py",
     '        report["diagnostics_run_id"] = self.run_id\n',
     f'        report["diagnostics_run_id"] = {HIT} and None\n',
     "test_cd08_tree_report_carries_unused", "CD07_STAMP"),
    ("W07_cd07_worker_drops_run_id", "train_single_trial.py",
     "            'run_id': args.diagnostics_run_id,\n",
     f"            'run_id': {HIT} and None,\n",
     "test_cd07_worker_cli_stamps_run_id", "CD07_WORKER"),
    ("W08_cd07_legacy_caller_ignored", "training_health_check.py",
     "    if run_ctx.get('expected_run_id') or run_ctx.get('run_started_at'):\n",
     f"    if {HIT} and False:\n",
     "test_cd07_legacy_path_only_with_caller_run", "CD07_LEGACY_CALLER"),
    ("W09_cd08_metric_not_computed", "training_diagnostics.py",
     '            report["unused_feature_pct"] = report["unused_features"]["pct"]\n',
     f'            report.pop("unused_features") if {HIT} else None\n',
     "test_cd08_tree_report_carries_unused", "CD08_REPORT"),
    ("W10_cd08_never_graded", "training_health_check.py",
     "    if settings.get('unused_feature_grading', DEFAULT_UNUSED_FEATURE_GRADING) != 'off':\n",
     f"    if {HIT} and False:\n",
     "test_cd08_fixed_grading[all-56.2-warning]", "CD08_GRADE"),
    ("W11_cd08_constant_not_excluded", "training_diagnostics.py",
     '    keep = [not (by_feature and by_feature.get(nm) in ("constant", "run")) for nm in names]\n',
     f'    keep = [{HIT} for nm in names]\n',
     "test_cd08_unused_metric_and_classes", "CD08_METRIC"),
    ("W12_cd09_ignore_expiry", "training_health_check.py",
     "    return now < expires, info\n",
     f"    return {HIT}, info\n",
     "test_cd09_expiry_clock", "CD09_EXPIRY"),
    ("W13_cd09_mpo_old_rule", "meta_prediction_optimizer_anti_overfit.py",
     "                if (_skip_active(_entry, _skip_rules)[0] if _skip_active is not None\n",
     f"                if ({HIT} and _entry.get('consecutive_critical', 0) >= _skip_threshold if _skip_active is not None\n",
     "test_cd09_both_readers[hours]", "CD09_MPO"),
    ("W14_cd09_thc_old_rule", "training_health_check.py",
     "        return skip_active(entry, skip_rules)[0]  # [CH14-DIAG CD-09] expiry_mode\n",
     f"        return {HIT} and entry.get('consecutive_critical', 0) >= skip_rules.get('max_consecutive_critical', 3)\n",
     "test_cd09_both_readers[hours]", "CD09_THC"),
    ("W15_cd10_no_prune_call", "training_health_check.py",
     "        _prune_health_history(_load_policies())\n",
     f"        {HIT}\n",
     "test_cd10_prune[move]", "CD10_PRUNE"),
    ("W16_cd10_root_cause_not_excluded", "training_health_check.py",
     "    exclude = HISTORY_OTHER_WRITER_PREFIXES + (() if st['history_prune_include_root_cause']\n",
     f"    exclude = HISTORY_OTHER_WRITER_PREFIXES + (() if {HIT} or st['history_prune_include_root_cause']\n",
     "test_cd10_prune[move]", "CD10_PRUNE"),
    ("W17_cd10_move_deletes", "training_diagnostics.py",
     '        if mode == "move":\n            dest',
     f'        if {HIT} and False:\n            dest',
     "test_cd10_prune[move]", "CD10_MODE"),
    ("W18_cd11_writer_off", "training_health_check.py",
     "        if _wave2_settings(policies)['training_health_log'] != 'on':\n",
     f"        if {HIT}:\n",
     "test_cd11_log_writer_and_reader", "CD11_LOG"),
    ("W19_cd11_reader_oldest_first", "parameter_advisor.py",
     "        return list(reversed(rows[-n:])) if n > 0 else []\n",
     f"        return list(rows[:n]) if {HIT} else []\n",
     "test_cd11_log_writer_and_reader", "CD11_READER"),
    ("W20_cd11_gate_ignored", "parameter_advisor.py",
     "        if _reads_training_health(watcher_policies):\n            signals",
     f"        if {HIT} or _reads_training_health(watcher_policies):\n            signals",
     "test_cd11_advisor_gate", "CD11_DEFAULT_OFF"),
    ("W21_cd12_report_mode_uses_legacy", "diagnostics_llm_analyzer.py",
     '    if prompt_fields == "report":  # [CH14-DIAG CD-12]',
     f'    if {HIT} and False:  # [CH14-DIAG CD-12]',
     "test_cd12_prompt_report_fields", "CD12_PROMPT"),
    ("W22_cd12_no_class_tag", "diagnostics_llm_analyzer.py",
     '        out.append({k: e.get(k) for k in ("feature", "importance", "magnitude", "column_class")\n',
     f'        out.append({{k: e.get(k) for k in ("feature", "importance", "magnitude") if {HIT}}} or {{k: e.get(k) for k in ("feature", "importance", "magnitude")\n',
     "test_cd12_prompt_report_fields", "CD12_PROMPT"),
    ("W23_cd12_default_reads_report", "diagnostics_llm_analyzer.py",
     'DEFAULT_LLM_PROMPT_FIELDS = "legacy"\n',
     f'DEFAULT_LLM_PROMPT_FIELDS = {HIT} and "report"\n',
     "test_cd12_default_prompt_identical_to_base", "CD12_DEFAULT"),
    ("W24_cd23_old_definition", "training_diagnostics.py",
     '        if self.dead_unit_definition == "always_negative_preact" and self._dead_unit_history:\n',
     f'        if {HIT} and False:\n',
     "test_cd23_dead_unit_metric[always_negative_preact-True-33.333333333333336]", "CD23_FIRES"),
    ("W25_cd23_probe_in_train_mode", "training_diagnostics.py",
     "            model.eval()\n            x = X_probe.detach()",
     f"            {HIT}; model.train()\n            x = X_probe.detach()",
     "test_cd23_cd07_nn_bit_identical[full]", "CD23_BITID"),
    ("W26_ch14_4_manifest_ignored", "training_diagnostics.py",
     "    return classify_columns(X, feature_names, run_level_names=names, run_level_source=src,\n",
     f"    return classify_columns(X, feature_names, run_level_names=None if {HIT} else names, run_level_source=src,\n",
     "test_ch14_4_run_class_from_manifest", "CH14_4_RUN"),
    ("W27_ch14_4_wrong_alias", "training_diagnostics.py",
     'FEATURE_MANIFEST_RUN_ALIAS = "run_context"\n',
     f'FEATURE_MANIFEST_RUN_ALIAS = {HIT} and "survivor_local"\n',
     "test_ch14_4_real_learn_manifest_read_only", "CH14_4_REAL"),
    ("W28_run_record_report_dropped", "training_diagnostics.py",
     '        report["diagnostics_settings"] = self._settings_record()\n',
     f'        report["diagnostics_settings"] = {HIT} and {{}}\n',
     "test_cd23_cd07_nn_bit_identical[full]", "RUN_RECORD_REPORT"),
    ("W29_run_record_log_dropped", "training_health_check.py",
     "            'settings': {**_grading_settings(policies), **_wave2_settings(policies)},\n",
     f"            'settings': {HIT} and {{}},\n",
     "test_cd11_log_writer_and_reader", "RUN_RECORD_LOG"),
    # landing re-fit 2026-11-25: one mutant for each gate that had none
    ("W30_cd07_bundle_no_generated_id", "training_diagnostics.py",
     "        run_id, gen_started = new_run_id()\n",
     f"        run_id, gen_started = ({HIT} and None), new_run_id()[1]\n",
     "test_cd07_options_bundle_and_child_flags", "CD07_BUNDLE"),
    ("W31_cd07_stale_action_ignored", "training_health_check.py",
     "    action = _wave2_settings(policies)['stale_report_action']\n",
     f"    action = {HIT} and 'PROCEED_WITH_NOTE'\n",
     "test_cd07_stale_action_setting[RETRY]", "CD07_STALE_ACTION"),
    ("W32_cd07_unknown_value_accepted", "training_health_check.py",
     "        'stale_report_action': _choice(policies, 'stale_report_action', STALE_REPORT_ACTIONS,\n                                       DEFAULT_STALE_REPORT_ACTION),\n",
     f"        'stale_report_action': {HIT} and policies.get('stale_report_action', DEFAULT_STALE_REPORT_ACTION),\n",
     "test_cd07_unknown_setting_value_fails_closed", "CD07_FAIL_CLOSED"),
    ("W33_cd09_default_skip_status_drift", "training_health_check.py",
     "                'threshold': max_consecutive,\n",
     f"                'threshold': max_consecutive + ({HIT} and 1),\n",
     "test_cd09_skip_status_default_identical", "CD09_DEFAULT"),
    ("W34_cd10_fifo_default_mode_changed", "training_diagnostics.py",
     '        if mode not in ("delete", "move"):\n',
     f'        mode, archive_dir = ("move", (archive_dir or history_dir + "_mut")) if {HIT} else (mode, archive_dir)\n        if mode not in ("delete", "move"):\n',
     "test_cd10_td_fifo_defaults_unchanged", "CD10_DEFAULT"),
    ("W35_cd12_no_old_archive_fallback", "diagnostics_llm_analyzer.py",
     '    rounds = ts.get("rounds_captured", (diag.get("training_rounds") or {}).get("total"))\n',
     f'    rounds = ts.get("rounds_captured") if {HIT} else None\n',
     "test_cd12_old_archive_fallback", "CD12_FALLBACK"),
    ("W36_cd23_effective_reports_requested", "training_diagnostics.py",
     '            model_specific["gradient_health"]["dead_unit_definition_effective"] = _dead_used\n',
     f'            model_specific["gradient_health"]["dead_unit_definition_effective"] = {HIT} and self.dead_unit_definition\n',
     "test_cd23_without_probe_says_old_definition_used", "CD23_EFFECTIVE"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    if a.jobs > 2:
        raise SystemExit("at most 2 jobs (lead's load note 2026-11-23 19:10)")
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
        results = list(ex.map(lambda m: w1.run_one(m, base, basetemp), todo))
    summary = {"schema": "ch14_wave2_mutants/1.0", "jobs": a.jobs, "wall_seconds": round(time.time() - t0, 1),
               "total": len(results), "killed": sum(r["killed"] for r in results), "results": results}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    for r in results:
        print(f"{'KILLED ' if r['killed'] else 'SURVIVED'} {r['id']:40s} rc={r['pytest_rc']} hits={r['marker_hits']} "
              f"tag={r['credited_tag_in_failure']} {r['seconds']}s")
    print(f"{summary['killed']}/{summary['total']} killed in {summary['wall_seconds']}s (jobs={a.jobs})")
    return 0 if summary["killed"] == summary["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
