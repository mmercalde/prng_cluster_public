#!/usr/bin/env python3
"""
why_report_auto.py - unit W2 (2026-11-27): WATCHER starts the why-report by itself at the END of a pipeline run.
Plan PLAN_LAND_LL7_LL8_20261126 §6.3 (W2 + W2 switch), lead decision Q-C (end of run, never delays Step 6).

WATCHER (agents/watcher_agent.py, owner lands by hand) makes ONE guarded call, only after a run that reached its
normal end (third pass, lead decision 1):  auto_run(logger).  All logic lives here so the owner patch stays small.

Switch (watcher_policies.json -> training_diagnostics):
  why_report_auto        "off" (default; also when the key, the section or the file is missing) | "on"
  why_report_wf_roots    list of walk-forward run dirs (or origin dirs) the runner searches; default [];
                         with the switch on it must name at least one (empty or missing = a bad value)
  why_report_timeout_s   number > 0, the runner's time limit PER FLAG; default 1800
Off: returns at once; no child is started and nothing is written.
A bad value (switch not off/on, roots not a list of strings or empty with the switch on, timeout not a number > 0,
section not an object, unreadable file): logged at ERROR, no child is started.
On (third pass, lead decision 2, plan §6.3 process boundary): the runner is NOT loaded into WATCHER. ONE child
process is started, in its own session:
    python3 -B scripts/diagnostics/why_report_runner.py --all-open --json --wf-root R ... --timeout-s T
            [--root ROOT] [--out-root OUT]
with an overall deadline of (open flags x T) + DEADLINE_MARGIN_S. On expiry the whole process group is killed and
reaped. Its stdout (the JSON result) and stderr go to files under <root or this tree>/data/why_reports/watcher_auto/
(or under --out-root when one is given). The JSON list is read back:
  - a failed attempt: ERROR log line naming the reason (never "None": "reason not given" when missing);
  - status ok with attempt_record_error: the report WAS attached; the alert says so and says that writing the
    attempt record failed, with the error (third pass, lead decision 3);
  - when any result needs attention, ONE notification lists them (why_report_flags.send_notification).
A child that dies, times out or prints no usable JSON: ONE ERROR log line and ONE notification that say so.
Never raises an Exception (everything inside try/except Exception -> logger.error(..., exc_info=True)), so the
pipeline's result cannot change. SystemExit and KeyboardInterrupt are not caught; the child is still killed on
the way out (finally). Advisory: changes no setting.

Kept out of why_report_flags.py on purpose: that module's return values are pinned to earlier revisions by
identity gates and 25 mutants (tests/test_why_report_flags.py, tests/why_flags_mutants.py).
"""
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

POLICIES_PATH = "watcher_policies.json"
SECTION = "training_diagnostics"
SWITCH_KEY, ROOTS_KEY, TIMEOUT_KEY = "why_report_auto", "why_report_wf_roots", "why_report_timeout_s"
SWITCH_VALUES = ("off", "on")
DEFAULT_SWITCH, DEFAULT_ROOTS, DEFAULT_TIMEOUT_S = "off", [], 1800
HERE = Path(__file__).resolve().parent
RUNNER_PATH = HERE / "scripts" / "diagnostics" / "why_report_runner.py"
LOG_SUBDIR = Path("data") / "why_reports" / "watcher_auto"
DEADLINE_MARGIN_S = 300          # on top of (open flags x why_report_timeout_s): runner start-up, reads, attach
TAG = "[WATCHER][WHY-REPORT]"


def read_settings(root=None):
    """(mode, roots, timeout_s, error). mode None with error set = a bad value (no child is started)."""
    p = os.path.join(root, POLICIES_PATH) if root else POLICIES_PATH
    if not os.path.isfile(p):
        return DEFAULT_SWITCH, list(DEFAULT_ROOTS), DEFAULT_TIMEOUT_S, None
    try:
        with open(p) as f:
            pol = json.load(f)
    except (OSError, ValueError) as e:
        return None, None, None, f"{p} unreadable: {type(e).__name__}: {e}"
    if not isinstance(pol, dict):
        return None, None, None, f"{p} must hold a JSON object, got {type(pol).__name__}"
    td = pol.get(SECTION)
    if td is None:
        td = {}
    if not isinstance(td, dict):
        return None, None, None, f"{SECTION} must be an object, got {type(td).__name__}"
    mode = td.get(SWITCH_KEY, DEFAULT_SWITCH)
    if mode not in SWITCH_VALUES:
        return None, None, None, f"{SECTION}.{SWITCH_KEY} must be one of {SWITCH_VALUES}, got {mode!r}"
    if mode == "off":
        return "off", None, None, None
    roots = td.get(ROOTS_KEY, list(DEFAULT_ROOTS))
    if not isinstance(roots, list) or not all(isinstance(r, str) and r for r in roots):
        return None, None, None, f"{SECTION}.{ROOTS_KEY} must be a list of paths, got {roots!r}"
    if not roots:   # [second pass, lead decision 4] on with no root: every flag would fail no_graded_block
        return None, None, None, (f"{SECTION}.{ROOTS_KEY} is empty (or missing) with {SWITCH_KEY} on; "
                                  f"name at least one walk-forward root")
    t = td.get(TIMEOUT_KEY, DEFAULT_TIMEOUT_S)
    if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t <= 0:
        return None, None, None, f"{SECTION}.{TIMEOUT_KEY} must be a number > 0, got {t!r}"
    return "on", list(roots), t, None


def runner_argv(roots, timeout_s, root=None, out_root=None):
    """The one child command (lead decision 2)."""
    argv = [sys.executable, "-B", str(RUNNER_PATH), "--all-open", "--json"]
    for r in roots:
        argv += ["--wf-root", r]
    argv += ["--timeout-s", repr(float(timeout_s))]
    if root:
        argv += ["--root", str(root)]
    if out_root:
        argv += ["--out-root", str(out_root)]
    return argv


def _kill_group(proc):
    """Kill the child's whole process group (it runs in its own session) and reap the child."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        proc.wait(timeout=30)
    except Exception:
        pass


def run_child(roots, timeout_s, n_open, root=None, out_root=None):
    """Start the runner child, wait up to the deadline, read its JSON. Returns
    (results or None, problem or None, info dict). problem is a plain sentence when the child died, timed out or
    printed no usable JSON."""
    base = Path(out_root) if out_root else Path(root or os.getcwd()) / LOG_SUBDIR
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    logdir = base / f"auto_{stamp}"
    logdir.mkdir(parents=True, exist_ok=False)
    out_p, err_p = logdir / "runner_stdout.json", logdir / "runner_stderr.log"
    deadline_s = max(1, n_open) * float(timeout_s) + DEADLINE_MARGIN_S
    argv = runner_argv(roots, timeout_s, root, out_root)
    info = {"argv": argv, "deadline_s": deadline_s, "stdout": str(out_p), "stderr": str(err_p), "returncode": None}
    timed_out = False
    with open(out_p, "wb") as fo, open(err_p, "wb") as fe:
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=fo, stderr=fe, cwd=str(root) if root else None,
                                start_new_session=True)
        try:
            try:
                proc.wait(timeout=deadline_s)
            except subprocess.TimeoutExpired:
                timed_out = True
        finally:   # whatever happens (also SystemExit / KeyboardInterrupt): no child or grandchild is left behind
            _kill_group(proc)
    info["returncode"] = proc.returncode
    if timed_out:
        return None, (f"the why-report runner did not finish within {deadline_s:.0f} s "
                      f"({n_open} open flag(s) x {timeout_s} s + {DEADLINE_MARGIN_S} s); its process group was "
                      f"killed"), info
    try:
        results = json.loads(out_p.read_text() or "null")
    except ValueError as e:
        results, bad = None, f"{type(e).__name__}: {e}"
    else:
        bad = None if isinstance(results, list) and all(isinstance(r, dict) for r in results) else \
            f"not a JSON list of objects ({type(results).__name__})"
    if proc.returncode not in (0, 1):
        return None, (f"the why-report runner died or refused its arguments (exit status {proc.returncode}; "
                      f"0 or 1 expected)"), info
    if bad:
        return None, f"the why-report runner printed no usable JSON ({bad}; exit status {proc.returncode})", info
    return results, None, info


def _what(r):
    """One plain line per result that needs attention. Never says 'reason None'."""
    fid = r.get("flag_id")
    if r.get("status") == "ok" and r.get("attempt_record_error"):
        return (f"{fid}: report ATTACHED ({r.get('report_path')}); writing the attempt record FAILED: "
                f"{r.get('attempt_record_error')}")
    reason = r.get("reason") or "reason not given"
    tail = f"; the attempt record was NOT written: {r['attempt_record_error']}" if r.get("attempt_record_error") else ""
    return f"{fid}: why-report FAILED ({reason}){tail}"


def auto_run(logger, root=None, out_root=None):
    """WATCHER's end-of-run hook. Returns the runner's results (list), or None when off, refused or the child
    failed. Never raises an Exception."""
    try:
        mode, roots, timeout_s, err = read_settings(root)
        if err:
            logger.error("%s %s; automatic why-report NOT run", TAG, err)
            return None
        if mode == "off":
            return None
        from why_report_flags import open_flags, send_notification
        n_open = len(open_flags(root))
        if n_open == 0:
            logger.info("%s no open why-report flag; runner not started", TAG)
            return []
        t0 = time.time()
        results, problem, info = run_child(roots, timeout_s, n_open, root=root, out_root=out_root)
        if problem:
            logger.error("%s automatic why-report NOT completed: %s; stdout %s, stderr %s", TAG, problem,
                         info["stdout"], info["stderr"])
            ok, detail = send_notification(f"{TAG}\nAutomatic why-report NOT completed: {problem}.\n"
                                           f"Log: {info['stderr']}", root)
            (logger.warning if ok else logger.error)("%s failure notification %s: %s", TAG,
                                                     "sent" if ok else "NOT sent", detail)
            return None
        attention = []
        for r in results:
            if r.get("status") == "failed" or r.get("attempt_record_error"):
                attention.append(r)
                logger.error("%s %s%s%s%s", TAG, _what(r),
                             f" ({r['error_class']})" if r.get("error_class") else "",
                             f": {r['message']}" if r.get("message") else "",
                             f"; log {r['log_path']}" if r.get("log_path") else "")
            elif r.get("status") == "ok":
                logger.warning("%s %s: why-report attached: %s", TAG, r.get("flag_id"), r.get("report_path"))
            else:
                logger.info("%s %s: %s (%s)", TAG, r.get("flag_id"), r.get("status"), r.get("message"))
        logger.info("%s runner child: %d result(s) in %.1f s, exit status %s; stderr %s", TAG, len(results),
                    time.time() - t0, info["returncode"], info["stderr"])
        if attention:
            ok, detail = send_notification(
                f"{TAG}\nAutomatic why-report: {len(attention)} of {len(results)} flag(s) need attention:\n"
                + "\n".join(_what(r) for r in attention[:5]) + ("\n..." if len(attention) > 5 else ""), root)
            (logger.warning if ok else logger.error)("%s notification %s: %s", TAG,
                                                     "sent" if ok else "NOT sent", detail)
        return results
    except Exception as e:  # never changes the pipeline's result; logged with its traceback, not swallowed
        logger.error("%s automatic why-report step failed (pipeline result unchanged): %r", TAG, e, exc_info=True)
        return None
