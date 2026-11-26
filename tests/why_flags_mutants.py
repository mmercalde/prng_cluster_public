#!/usr/bin/env python3
"""
Mutants for tests/test_why_report_flags.py (2026-11-26). Each mutant is one text change to a COPY of
why_report_flags.py, training_health_check.py or agents/watcher_agent.py, made under --out (never the
worktree's own files). The named test runs against the copy (WHY_CODE_ROOT) and must FAIL with the named
[CREDIT:<tag>] in its output. The unmutated copy must pass the whole suite first. CPU only; one pytest at
a time; no notification is ever sent (the suite's autouse fixture fakes why_report_flags.subprocess and
WATCHER's notify_telegram is replaced by a list).

    PYTHONDONTWRITEBYTECODE=1 nice -n 15 python3 -B -u tests/why_flags_mutants.py --out data/why_flags/mutants
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[1]
SUITE = "tests/test_why_report_flags.py"
WRF, THC, WA = "why_report_flags.py", "training_health_check.py", "agents/watcher_agent.py"
FILES = (WRF, THC, WA)
T_START = "test_run_start_lists_open_flags_and_reminds_once"
T_NEW = "test_new_flag_notified_once_repeat_not"

MUTANTS = [
    # (id, file, old, new, test node, credit tag)
    ("WM1_no_open_flag_listing", WRF,
     '        print("\\n".join(lines), flush=True)\n        for l in lines:\n            log.warning(l)\n',
     "        pass\n", T_START, "WSTART"),
    # [R1] anchors moved with the own sender (F2)
    ("WM2_no_new_flag_notify", WRF, "    ok, detail = _send(notify, message, root)\n",
     '    ok, detail = True, "mutant: nothing sent"\n', T_NEW, "WNEW"),
    ("WM3_no_run_reminder", WRF, '            ok, detail = _send(notify, f"{TAG}\\nReminder:',
     '            ok, detail = (lambda *a: (True, ""))(notify, f"{TAG}\\nReminder:', T_START, "WREMIND"),
    ("WM4_reader_failure_raises", WRF,
     '        log.error("%s flag reader failed at run start (pipeline continues): %r", TAG, e, exc_info=True)\n'
     '        return []\n',
     "        raise\n", "test_reader_failure_never_stops_the_pipeline", "WFAIL"),
    ("WM5_watcher_no_run_start_call", WA,
     "        _why_run_start(logger)  # [BENCHING] open why-report flags, top of summary and log\n",
     "", T_START, "WSTART"),
    ("WM6_watcher_no_health_call", WA,
     "        _why_on_health(health, logger)  # [BENCHING] one notification per new flag_id (own sender, exit checked)\n",
     "", T_NEW, "WNEW"),
    ("WM7_notify_every_repeat", WRF, "        if fid in sent:\n", "        if False:\n", T_NEW, "WNEW"),
    ("WM8_output_with_no_flags", WRF, "        if not lines:\n            return []\n",
     '        if not lines:\n            print("", flush=True)\n            return []\n',
     "test_no_flags_identical_to_base", "WIDENT"),
    ("WM9_gitignore_dropped", ".gitignore", "diagnostics_outputs/why_report_notified.jsonl\n", "",
     "test_state_files_are_git_ignored", "GITIGNORE"),
    ("WM10_reader_ignores_attach", WRF, '        g["open"] = not g["attachments"]\n', '        g["open"] = True\n',
     "test_attach_closes_and_is_append_only", "ATTACH"),
    # [R1 F2] Beta Q6: the sender checks the exit status, a failure is recorded as failed and tried again
    ("WM11_exit_status_ignored", WRF, "        if p.returncode != 0:\n", "        if False:\n",
     "test_sender_outcomes[exit-False-exited 3: synthetic failure]", "WSEND"),
    ("WM12_timeout_not_named", WRF,
     '    except subprocess.TimeoutExpired:\n        return False, f"notification script timed out after {timeout:g}s"\n', "",
     "test_sender_outcomes[timeout-False-timed out]", "WSEND"),
    ("WM13_no_timeout", WRF, "stderr=subprocess.PIPE, timeout=timeout)", "stderr=subprocess.PIPE)",
     "test_sender_outcomes[ok-True-exited 0]", "WSEND"),
    ("WM14_policy_path_ignored", WRF, '        script = n.get("script_path") or script\n', "        pass\n",
     "test_sender_takes_the_policy_script", "WSEND"),
    ("WM15_failure_recorded_as_sent", WRF, '"status": "sent" if ok else "failed"', '"status": "sent"',
     "test_failed_send_logged_recorded_and_retried[same_flag_health-spawn]", "WRETRY"),
    ("WM16_no_retry", WRF, "    pending = failed - sent - {skip}\n", "    pending = set()\n",
     "test_failed_send_logged_recorded_and_retried[other_health-exit]", "WRETRY"),
    ("WM17_failure_logged_as_sent", WRF,
     '        log.error("%s NEW flag %s: notification NOT sent (%s); recorded',
     '        log.warning("%s NEW flag %s: notification sent (%s); recorded',
     "test_failed_send_logged_recorded_and_retried[same_flag_health-timeout]", "WLOG"),
    ("WM18_watcher_uses_spawn_and_forget", WA,
     "        _why_on_health(health, logger)  # [BENCHING] one notification per new flag_id (own sender, exit checked)\n",
     "        _why_on_health(health, logger, lambda m: (notify_telegram(m), True)[1])\n",
     "test_watcher_send_failure_then_retry", "WLOG"),
    # [R1 F1] flag_id_of and the writer agree for a line without flag_id
    ("WM19_reader_formula_no_first_at", WRF,
     "flag.get('critical_since') or flag.get('why_flag_first_at') or flag.get('flagged_at')",
     "flag.get('critical_since') or flag.get('flagged_at')",
     "test_flag_id_formula_agrees_with_writer[unknown]", "FIDFORMULA"),
    ("WM20_writer_drops_first_at", THC, "        'why_flag_first_at': entry.get('why_flag_first_at'),\n", "",
     "test_flag_id_formula_agrees_with_writer[unknown]", "FIDFORMULA"),
    # [R1 F3] a NEW linked flag after an attached why-report, and none before max_consecutive more
    ("WM21_no_new_segment", THC,
     "            and _why_flag_attached(why_flag_id(model_type, _why_flag_anchor(entry))):\n",
     # without the new segment the attached flag keeps being re-raised: the "fewer than three" check goes red
     "            and False:\n", "test_new_flag_after_attached_report_when_still_critical", "REFLAG_QUIET"),
    ("WM22_no_segment_guard", THC,
     "            if consecutive - entry.get('why_flag_base', 0) < max_consecutive:\n", "            if False:\n",
     "test_new_flag_after_attached_report_when_still_critical", "REFLAG_QUIET"),
    ("WM23_no_previous_link", THC, "        'previous_flag_id': entry.get('why_flag_prev'),\n",
     "        'previous_flag_id': None,\n", "test_new_flag_after_attached_report_when_still_critical", "REFLAG"),
]


def run_pytest(code_root, node, basetemp, log, extra_env=None):
    env = dict(os.environ, WHY_CODE_ROOT=str(code_root), CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    env.update(extra_env or {})
    cmd = [sys.executable, "-m", "pytest", "-q", f"{SUITE}::{node}" if node else SUITE,
           f"--basetemp={basetemp}", "-p", "no:cacheprovider"]
    p = subprocess.run(cmd, cwd=str(WORKTREE), env=env, capture_output=True, text=True, timeout=1800)
    Path(log).write_text(p.stdout + "\n--- stderr ---\n" + p.stderr)
    return p.returncode, p.stdout + p.stderr


def _gitignore_mutant(root, out, mid, old, new, node, tag):
    """The .gitignore mutant runs `git check-ignore` against a scratch repository holding the mutated file
    (the worktree's own .gitignore is never touched)."""
    repo = root / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    txt = (WORKTREE / ".gitignore").read_text()
    if txt.count(old) != 1:
        return None
    (repo / ".gitignore").write_text(txt.replace(old, new))
    fails = []
    from importlib import util
    spec = util.spec_from_file_location("wrf_m", str(WORKTREE / WRF))
    m = util.module_from_spec(spec)
    spec.loader.exec_module(m)
    for p in (m.FLAGS_PATH, m.ATTACH_PATH, m.NOTIFIED_PATH):
        r = subprocess.run(["git", "-C", str(repo), "check-ignore", "-q", "--no-index", p])
        if r.returncode != 0:
            fails.append(p)
    txt = f"[CREDIT:{tag}] not ignored: {fails}" if fails else "all ignored"
    (out / f"{mid}.log").write_text(txt)
    return (1 if fails else 0), txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if out.exists():
        sys.exit(f"--out {out} exists; give a new directory")
    out.mkdir(parents=True)

    def copy_tree(root):
        for f in FILES:
            (root / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(WORKTREE / f, root / f)

    results = []
    clean = out / "clean"
    copy_tree(clean)
    rc, _ = run_pytest(clean, None, out / "clean_pt", out / "clean.log")
    print(f"[{'PASS' if rc == 0 else 'FAIL'}] unmutated copy passes the suite (rc={rc})", flush=True)
    results.append({"mutant": "clean", "ok": rc == 0})
    for mid, fn, old, new, node, tag in MUTANTS:
        if a.only and mid not in a.only:
            continue
        root = out / mid
        copy_tree(root)
        if fn == ".gitignore":
            r = _gitignore_mutant(root, out, mid, old, new, node, tag)
            if r is None:
                print(f"[FAIL] {mid}: anchor not found once", flush=True)
                results.append({"mutant": mid, "ok": False, "why": "anchor"})
                continue
            rc, txt = r
        else:
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
