#!/usr/bin/env python3
"""
Mutants for tests/test_benching_switch.py (2026-11-26). Each mutant is one text change to a COPY of a reader
file, made under --out (never the worktree's own files). The named test is run against the copy
(BENCH_CODE_ROOT) and must FAIL with the named [CREDIT:<tag>] in its output. The unmutated copy must pass
the same tests first. CPU only; one pytest at a time.

    PYTHONDONTWRITEBYTECODE=1 nice -n 15 python3 -B -u tests/bench_switch_mutants.py --out data/bench_switch/mutants
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[1]
SUITE = "tests/test_benching_switch.py"
THC, MPO = "training_health_check.py", "meta_prediction_optimizer_anti_overfit.py"

MUTANTS = [
    # (id, file, old, new, test node, credit tag)
    ("M01_no_off_branch_registry", THC,
     # [R1] anchor moved: the off branch now also holds the F3 segment guard
     "        if bench == 'off':\n            if consecutive - entry.get('why_flag_base', 0) < max_consecutive:\n"
     "                return 'RETRY', consecutive, None   # [R1 F3] after an attached why-report, not yet again\n"
     # [W2 2026-11-27] anchor moved: the call now passes run_ctx (models_dir for the flag line)
     "            return 'RETRY', consecutive, _raise_why_flag(model_type, entry, max_consecutive, run_ctx)\n",
     "", "test_off_never_skipped_flag_raised[absent]", "OFF_THC"),
    ("M02_no_off_branch_skip_active", THC,
     "    if bench == 'off':\n        info['why_report_needed'] = True\n        return False, info\n",
     "", "test_off_compare_reader_keeps_family", "OFF_MPO"),
    ("M03_no_flag_in_result", THC,
     "        result['why_report_needed'] = why\n", "        pass\n",
     "test_off_never_skipped_flag_raised[absent]", "FLAG_RESULT"),
    ("M04_no_flag_file", THC,
     "            f.write(json.dumps(flag, sort_keys=True, default=str) + \"\\n\")\n", "            pass\n",
     "test_off_never_skipped_flag_raised[absent]", "FLAG_FILE"),
    ("M05_no_flag_log", THC,
     "    logger.warning(\n        \"[WHY-REPORT] Model", "    logger.info(\n        \"[WHY-REPORT] Model",
     "test_off_never_skipped_flag_raised[absent]", "FLAG_LOG"),
    ("M06_no_hours_restart", THC,
     # [PACKAGE 2026-11-26] anchor moved: the restart now also clears the flag identity keys
     "        entry['consecutive_critical'] = 0\n        for k in _WHY_FLAG_ENTRY_KEYS:",
     "        pass\n        for k in _WHY_FLAG_ENTRY_KEYS:",
     "test_hours_restart[25-expect0]", "HOURS_RESTART"),
    ("M07_restart_always", THC,
     "    if bench == 'on' and entry.get('consecutive_critical', 0) >= max_consecutive \\\n"
     "            and not skip_active(entry, skip_rules)[0]:",
     "    if bench == 'on' and entry.get('consecutive_critical', 0) >= max_consecutive:",
     "test_on_until_reset_equals_base", "ON_IDENTITY"),
    ("M08_default_on", THC, 'DEFAULT_BENCHING = "off"', 'DEFAULT_BENCHING = "on"',
     "test_off_never_skipped_flag_raised[absent]", "OFF_THC"),
    ("M09_no_validation", THC, "    if v not in BENCHING_SWITCH:\n", "    if False:\n",
     "test_bad_value_raises", "BAD_VALUE"),
    ("M10_no_compare_why_print", MPO, "            if _why_models:\n", "            if False:\n",
     "test_off_compare_reader_keeps_family", "OFF_MPO_WHY"),
    # [PACKAGE 2026-11-26] review fixes (b), (c), (d)
    ("M11_since_known_always_true", THC, "        'critical_since_known': since is not None,\n",
     "        'critical_since_known': True,\n",
     "test_since_unknown_for_a_count_that_predates_the_record", "SINCE_KNOWN"),
    ("M12_flag_id_per_line", THC, "why_flag_id(model_type, _why_flag_anchor(entry) or flagged_at)",
     "why_flag_id(model_type, flagged_at)", "test_flag_id_stable_and_since_known", "FLAG_ID"),
    ("M13_no_first_flagged_anchor", THC, "        entry['why_flag_first_at'] = now_iso\n", "        pass\n",
     "test_since_unknown_for_a_count_that_predates_the_record", "FLAG_ID"),
    ("M14_reset_keeps_flag_identity", THC,
     "            for k in _WHY_FLAG_ENTRY_KEYS:  # [BENCHING] the run of criticals ended",
     "            for k in ():  # [BENCHING] the run of criticals ended",
     "test_flag_id_stable_and_since_known", "FLAG_ID_NEW_RUN"),
    ("M15_fallback_benches_when_off", MPO,
     '                    _active, _why = _over and _skip_bench == "on", _over and _skip_bench == "off"\n',
     "                    _active, _why = _over, False\n",
     "test_compare_reader_fallback_without_health_check[off]", "FALLBACK"),
    ("M16_g2_flag_on_winner", THC,   # [W2 2026-11-27] anchor moved: run_ctx passed on
     "        action, consecutive, why = _severity_to_action(severity, model_type, policies, run_ctx)\n    for m, v",
     "        action, consecutive, why = _severity_to_action(severity, winner_name, policies, run_ctx)\n    for m, v",
     "test_g2_promotion_plus_why_flag", "G2_FLAG"),
    ("M17_fixed_path_drops_flag", THC,
     # [R1] anchor moved: the F4 provenance block now sits between grade_note and this call
     "    _attach_skip_fields(result, action, consecutive, why)\n    logger.info(\"Training health check (fixed)",
     "    _attach_skip_fields(result, action, consecutive, None)\n    logger.info(\"Training health check (fixed)",
     "test_fixed_grading_at_limit_benching_off", "FIXED_FLAG"),
    # [R1 F1] Beta Q5: the registry is written through a temp file and os.replace (check and reset)
    ("M18_check_truncate_and_write", THC,
     "        _write_skip_registry(registry)\n    except (IOError, OSError) as e:\n        logger.warning(f\"Failed to write skip",
     "        with open(SKIP_REGISTRY_PATH, 'w') as f:\n            json.dump(registry, f, indent=2)\n"
     "    except (IOError, OSError) as e:\n        logger.warning(f\"Failed to write skip",
     "test_interrupted_registry_write_keeps_flag_identity", "ATOMIC"),
    ("M19_reset_truncate_and_write", THC,
     "            _write_skip_registry(registry)   # [R1 F1] temp file + os.replace\n",
     "            with open(SKIP_REGISTRY_PATH, 'w') as f:\n                json.dump(registry, f, indent=2)\n",
     "test_interrupted_registry_write_keeps_flag_identity", "ATOMIC_RESET"),
    # [R1 F5] Beta Q3: the compare reader's import-failure fallback honours hours expiry
    ("M20_fallback_ignores_hours", MPO,
     '                    if _active and _rules_fb.get("expiry_mode") == "hours":',
     "                    if False:",
     "test_compare_fallback_honours_hours_expiry[expired]", "FALLBACK_HOURS"),
]


def run_pytest(code_root, node, basetemp, log):
    env = dict(os.environ, BENCH_CODE_ROOT=str(code_root), CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    cmd = [sys.executable, "-m", "pytest", "-q", f"{SUITE}::{node}" if node else SUITE, "-k", "not zz",
           f"--basetemp={basetemp}", "-p", "no:cacheprovider"]
    p = subprocess.run(cmd, cwd=str(WORKTREE), env=env, capture_output=True, text=True, timeout=1800)
    Path(log).write_text(p.stdout + "\n--- stderr ---\n" + p.stderr)
    return p.returncode, p.stdout + p.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if out.exists():
        sys.exit(f"--out {out} exists; give a new directory")
    out.mkdir(parents=True)
    results = []
    clean = out / "clean"
    clean.mkdir()
    for f in (THC, MPO):
        shutil.copy2(WORKTREE / f, clean / f)
    rc, _ = run_pytest(clean, None, out / "clean_pt", out / "clean.log")
    print(f"[{'PASS' if rc == 0 else 'FAIL'}] unmutated copy passes the suite (rc={rc})", flush=True)
    results.append({"mutant": "clean", "ok": rc == 0})
    for mid, fn, old, new, node, tag in MUTANTS:
        if a.only and mid not in a.only:
            continue
        root = out / mid
        root.mkdir()
        for f in (THC, MPO):
            shutil.copy2(WORKTREE / f, root / f)
        src = (root / fn).read_text()
        n = src.count(old)
        if n != 1:
            print(f"[FAIL] {mid}: anchor found {n} times (must be 1)", flush=True)
            results.append({"mutant": mid, "ok": False, "why": f"anchor x{n}"})
            continue
        (root / fn).write_text(src.replace(old, new))
        rc, txt = run_pytest(root, node, out / f"{mid}_pt", out / f"{mid}.log")
        killed = rc != 0 and f"[CREDIT:{tag}]" in txt
        print(f"[{'KILLED' if killed else 'SURVIVED'}] {mid}: {node} -> {tag} (rc={rc})", flush=True)
        results.append({"mutant": mid, "test": node, "tag": tag, "rc": rc, "ok": killed})
    (out / "mutants_summary.json").write_text(json.dumps(results, indent=2))
    ok = sum(r["ok"] for r in results)
    print(f"{ok}/{len(results)} ok (clean pass + mutants killed)", flush=True)
    sys.exit(0 if ok == len(results) else 1)


if __name__ == "__main__":
    main()
