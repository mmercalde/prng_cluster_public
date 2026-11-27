#!/usr/bin/env python3
"""
Why-report flags: the reader, the attach record, and WATCHER's look-out (2026-11-26).

Owner rulings 2026-11-26 07:05-07:10 (logs/e2e_session/OWNER_RULING_benching_switch_20261126.txt):
with benching off (training_health_check.DEFAULT_BENCHING) a model family at or over
max_consecutive_critical is never skipped; training_health_check appends a why-report flag line to
diagnostics_outputs/why_report_flags.jsonl. "I dont just want a flagged file that is going to be
forgotten - something needs to look out for this - something in the pipeline" (07:10). Stage 1, this
module: WATCHER lists the open flags at the start of every run and notifies the owner. Stage 2 (WATCHER
starts the why-report itself) is unit W of the LL-7/LL-8 plan, TODO-WHY-1, and is NOT built here.

Records (all append-only, all git-ignored, .gitignore "why-report flags"; nothing is ever rewritten):
  FLAGS_PATH        written by training_health_check._raise_why_flag (one line per critical at or over
                    the limit, so repeats share a flag_id)
  ATTACH_PATH       written by attach(): one line per why-report attached to a flag_id
  NOTIFIED_PATH     written by the hooks: one line per NEW-flag notification attempt, status 'sent' or
                    'failed' (a line with no status, from before R1, counts as sent)

A flag_id is open until a why-report is attached to it. Flag lines written before flag_id existed get
one derived the same way as training_health_check._raise_why_flag:
"<model_type>@<critical_since or why_flag_first_at or flagged_at>".
After a why-report is attached, a family that stays critical for another max_consecutive_critical checks
with no reset gets a NEW flag_id, linked by previous_flag_id (training_health_check, R1 F3).

Notifications (R1 F2, Beta X-BENCHING-SWITCH R1 Q6): send_notification runs the notification script
(watcher_policies.json notifications.script_path, default /usr/local/bin/cluster_notify.sh) with a short
timeout and checks its exit status. "notification sent" is logged only for exit status 0; a spawn error, a
non-zero exit or a timeout is logged at ERROR, recorded as a failed attempt, and the NEW-flag notification
is tried again at the next health check or run start. LIMIT: the installed cluster_notify.sh exits 0 even
when Telegram is unreachable (its header, "Silent on failure (exit 0 always)"), so exit 0 proves the
script ran, not that the message was delivered. WATCHER's own notify_telegram is not used here.

CLI:
    python3 why_report_flags.py list [--all] [--json]
    python3 why_report_flags.py attach FLAG_ID REPORT_PATH --by NAME [--note TEXT]

The WATCHER hooks (watcher_run_start, watcher_on_health) and send_notification never raise: a failure is logged at ERROR with
its traceback and the pipeline goes on.
"""
import argparse
import json
import logging
import os
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

FLAGS_PATH = "diagnostics_outputs/why_report_flags.jsonl"   # == training_health_check.WHY_REPORT_FLAGS_PATH
ATTACH_PATH = "diagnostics_outputs/why_report_attachments.jsonl"
ATTEMPTS_PATH = "diagnostics_outputs/why_report_attempts.jsonl"   # unit W1 (why_report_runner), append-only
ATTEMPT_SCHEMA = "ch14_why_report_attempt/1"
ATTEMPT_OUTCOMES = ("ok", "failed")
STATUS_OPEN, STATUS_FAILED, STATUS_CLOSED = "open", "why_report_failed", "closed"
NOTIFIED_PATH = "diagnostics_outputs/why_report_notified.jsonl"
ATTACH_SCHEMA = "ch14_why_report_attach/1"
NOTIFIED_SCHEMA = "ch14_why_report_notified/1"
TAG = "[WATCHER][WHY-REPORT]"
POLICIES_PATH = "watcher_policies.json"
DEFAULT_NOTIFY_SCRIPT = "/usr/local/bin/cluster_notify.sh"   # == agents/watcher_agent.py notify_telegram
NOTIFY_TIMEOUT_S = 15.0   # the installed script's curl gives up after --max-time 8

logger = logging.getLogger("why_report_flags")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path(root: Optional[str], rel: str) -> str:
    return os.path.join(root, rel) if root else rel


def _read_jsonl(path: str):
    """(records, bad_line_numbers). A missing file is no records. Bad lines are counted, never hidden."""
    if not os.path.isfile(path):
        return [], []
    recs, bad = [], []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                if not isinstance(r, dict):
                    raise ValueError("not an object")
                recs.append(r)
            except ValueError:
                bad.append(n)
    return recs, bad


def _append(path: str, rec: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")


def flag_id_of(flag: Dict[str, Any]) -> str:
    """The flag's flag_id, or the same id derived for a line written before flag_id existed."""
    fid = flag.get("flag_id")
    if fid:
        return str(fid)
    anchor = flag.get('critical_since') or flag.get('why_flag_first_at') or flag.get('flagged_at')
    return f"{flag.get('model_type', 'unknown')}@{anchor}"


def read_flags(root: Optional[str] = None) -> Dict[str, Any]:
    """
    Every flag grouped by flag_id, oldest first, with its attachments.
    Returns {'groups': OrderedDict[flag_id -> group], 'bad_lines': {file: [line numbers]}}.
    group: flag_id, model_type, first_flagged_at, last_flagged_at, repeats, max_consecutive_critical,
           threshold, critical_since, critical_since_known, attachments (list), open (bool).
    """
    flags, bad_f = _read_jsonl(_path(root, FLAGS_PATH))
    atts, bad_a = _read_jsonl(_path(root, ATTACH_PATH))
    groups: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for fl in flags:
        fid = flag_id_of(fl)
        g = groups.get(fid)
        if g is None:
            since = fl.get("critical_since")
            g = groups[fid] = {
                "flag_id": fid, "model_type": fl.get("model_type"),
                "first_flagged_at": fl.get("flagged_at"), "last_flagged_at": fl.get("flagged_at"),
                "repeats": 0, "max_consecutive_critical": fl.get("consecutive_critical"),
                "threshold": fl.get("threshold"), "critical_since": since,
                "critical_since_known": fl.get("critical_since_known", since is not None),
                "previous_flag_id": fl.get("previous_flag_id"),
                "attachments": [],
            }
        g["repeats"] += 1
        g["last_flagged_at"] = fl.get("flagged_at") or g["last_flagged_at"]
        cc = fl.get("consecutive_critical")
        if isinstance(cc, int) and (not isinstance(g["max_consecutive_critical"], int)
                                    or cc > g["max_consecutive_critical"]):
            g["max_consecutive_critical"] = cc
    orphan_atts = []
    for a in atts:
        g = groups.get(a.get("flag_id"))
        (g["attachments"] if g is not None else orphan_atts).append(a)
    for g in groups.values():
        g["open"] = not g["attachments"]
    bad = {k: v for k, v in ((FLAGS_PATH, bad_f), (ATTACH_PATH, bad_a)) if v}
    # [W1] attempts: only when the attempts file exists, so with no runner attempt every return value is
    # exactly the a85adb1 one (tests/test_why_report_runner.py proves it against git show a85adb1)
    ap = _path(root, ATTEMPTS_PATH)
    if os.path.isfile(ap):
        try:
            tries, bad_t = _read_jsonl(ap)
        except (OSError, ValueError) as e:
            # [W1 third pass, Beta X-W1-RUNNER R1 Q2] an unreadable attempts file never hides the flags or
            # their attachments: they are returned as without an attempts file, plus attempts_error
            logger.error("%s %s unreadable (%r): flags and attachments shown without runner attempts",
                         TAG, ATTEMPTS_PATH, e)
            return {"groups": groups, "bad_lines": bad, "orphan_attachments": orphan_atts,
                    "attempts_error": f"{ATTEMPTS_PATH}: {e!r}"}
        for g in groups.values():
            g["attempts"] = []
        for t in tries:
            g = groups.get(t.get("flag_id"))
            if g is not None:
                g["attempts"].append(t)
        for g in groups.values():
            g["attempt_count"] = len(g["attempts"])
            g["status"] = flag_status(g)
        if bad_t:
            bad[ATTEMPTS_PATH] = bad_t
    return {"groups": groups, "bad_lines": bad, "orphan_attachments": orphan_atts}


def flag_status(group: Dict[str, Any]) -> str:
    """[W1] 'closed' when a why-report is attached; 'why_report_failed' when open and the latest runner
    attempt failed; 'open' otherwise (no attempt, or the latest attempt did not fail)."""
    if group.get("attachments"):
        return STATUS_CLOSED
    tries = group.get("attempts") or []
    if tries and tries[-1].get("outcome") == "failed":
        return STATUS_FAILED
    return STATUS_OPEN


def record_attempt(flag_id: str, outcome: str, root: Optional[str] = None, reason: Optional[str] = None,
                   error_class: Optional[str] = None, message: Optional[str] = None,
                   log_path: Optional[str] = None, started_at: Optional[str] = None,
                   ended_at: Optional[str] = None, **extra: Any) -> Dict[str, Any]:
    """[W1] Append one runner attempt for flag_id to ATTEMPTS_PATH (schema ch14_why_report_attempt/1).
    The attempt number is the flag's earlier attempt count + 1. message is cut to its first 500 chars.
    Refuses an unknown flag_id, an outcome not in ATTEMPT_OUTCOMES, and a failed attempt with no reason.
    A failed attempt never attaches anything: the flag stays open, status why_report_failed."""
    if outcome not in ATTEMPT_OUTCOMES:
        raise ValueError(f"record_attempt: outcome {outcome!r} not in {ATTEMPT_OUTCOMES}")
    if outcome == "failed" and not reason:
        raise ValueError("record_attempt: a failed attempt needs a reason")
    rf = read_flags(root)
    if rf.get("attempts_error"):      # the attempt number would be wrong: refused, not guessed
        raise ValueError(f"record_attempt: {rf['attempts_error']}")
    groups = rf["groups"]
    if flag_id not in groups:
        raise ValueError(f"record_attempt: unknown flag_id {flag_id!r}")
    rec = {"schema": ATTEMPT_SCHEMA, "flag_id": flag_id,
           "attempt": int(groups[flag_id].get("attempt_count", 0)) + 1, "outcome": outcome,
           "reason": reason, "error_class": error_class,
           "message": None if message is None else str(message)[:500], "log_path": log_path,
           "started_at": started_at, "ended_at": ended_at or _now(), "recorded_at": _now()}
    for k, v in extra.items():
        rec.setdefault(k, v)
    _append(_path(root, ATTEMPTS_PATH), rec)
    return rec


def open_flags(root: Optional[str] = None) -> "OrderedDict[str, Dict[str, Any]]":
    """The open flags (no why-report attached), grouped by flag_id, oldest first."""
    return OrderedDict((k, g) for k, g in read_flags(root)["groups"].items() if g["open"])


def attach(flag_id: str, report_path: str, by: str, root: Optional[str] = None,
           note: Optional[str] = None, report_sha256: Optional[str] = None,
           extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Attach a why-report to a flag_id: one appended line in ATTACH_PATH. Refuses an unknown flag_id,
    a report file that does not exist, an empty `by`, or a flag_id that already has a report.
    [W1] report_sha256 (optional): refused unless it equals the report file's sha256; recorded. extra
    (optional): more keys recorded (never overriding the base keys). With neither, the record is the
    a85adb1 one."""
    if not by or not str(by).strip():
        raise ValueError("attach: --by is required (who attached the why-report)")
    groups = read_flags(root)["groups"]
    if flag_id not in groups:
        raise ValueError(f"attach: unknown flag_id {flag_id!r}; `list --all` shows the flag ids")
    if groups[flag_id]["attachments"]:
        raise ValueError(f"attach: {flag_id!r} already has a why-report: "
                         f"{groups[flag_id]['attachments'][0].get('report_path')}")
    rp = report_path if os.path.isabs(report_path) or not root else os.path.join(root, report_path)
    if not os.path.isfile(rp):
        raise ValueError(f"attach: report file not found: {report_path}")
    rec = {"schema": ATTACH_SCHEMA, "flag_id": flag_id, "report_path": report_path, "by": str(by),
           "attached_at": _now(), "note": note}
    if report_sha256 is not None:
        import hashlib
        with open(rp, "rb") as f:
            got = hashlib.sha256(f.read()).hexdigest()
        if got != report_sha256:
            raise ValueError(f"attach: report sha256 {got} != declared {report_sha256}")
        rec["report_sha256"] = report_sha256
    for k, v in (extra or {}).items():
        rec.setdefault(k, v)
    _append(_path(root, ATTACH_PATH), rec)
    return rec


def summary_lines(root: Optional[str] = None) -> List[str]:
    """Lines for the top of WATCHER's summary; empty when nothing is open and nothing is unreadable."""
    rf = read_flags(root)
    opened = [g for g in rf["groups"].values() if g["open"]]
    # [W1] flags whose latest runner attempt failed come FIRST, marked FAILED with the reason (a no-op
    # with no attempts file: no group has a status then, and the order is the a85adb1 order)
    failed = [g for g in opened if g.get("status") == STATUS_FAILED]
    opened = failed + [g for g in opened if g.get("status") != STATUS_FAILED]
    lines = []
    if opened:
        lines.append(f"{TAG} {len(opened)} open why-report flag(s) (benching off: families kept, "
                     f"a human or AI must check them):")
        for g in failed:
            t = g["attempts"][-1]
            lines.append(f"{TAG}   FAILED {g['flag_id']}: automatic why-report failed, reason {t.get('reason')} "
                         f"({t.get('error_class')}: {str(t.get('message'))[:200]}), attempt {t.get('attempt')} "
                         f"of {g['attempt_count']}, log {t.get('log_path')}")
        for g in opened:
            since = g["critical_since"] if g["critical_since_known"] else "unknown (count predates the record)"
            lines.append(f"{TAG}   {g['flag_id']}: {g['model_type']} critical {g['max_consecutive_critical']}x "
                         f"in a row (limit {g['threshold']}), since {since}, flagged {g['repeats']}x, "
                         f"last {g['last_flagged_at']}"
                         + (f", follows {g['previous_flag_id']} (why-report attached, still critical)"
                            if g.get('previous_flag_id') else ""))
        lines.append(f"{TAG}   attach a report: python3 why_report_flags.py attach FLAG_ID REPORT_PATH --by NAME")
    for fn, nums in rf["bad_lines"].items():
        lines.append(f"{TAG} {fn}: {len(nums)} unreadable line(s) (first at line {nums[0]}); not shown")
    if rf.get("attempts_error"):      # [W1] only when the attempts file exists and cannot be read
        lines.append(f"{TAG} ERROR attempts file unreadable, runner attempt history not shown: "
                     f"{rf['attempts_error']}")
    return lines


# ── Notification (R1 F2) ────────────────────────────────────────────────────

def _notify_settings(root: Optional[str]) -> Tuple[str, bool]:
    """(script path, telegram_enabled) from watcher_policies.json "notifications"; defaults when absent."""
    script, enabled = DEFAULT_NOTIFY_SCRIPT, True
    p = _path(root, POLICIES_PATH)
    if os.path.isfile(p):
        with open(p) as f:
            n = (json.load(f) or {}).get("notifications") or {}
        script = n.get("script_path") or script
        enabled = n.get("telegram_enabled", True) is not False
    return script, enabled


def send_notification(message: str, root: Optional[str] = None,
                      timeout: float = NOTIFY_TIMEOUT_S) -> Tuple[bool, str]:
    """Run the notification script and wait for it, at most `timeout` seconds. Returns (ok, detail);
    ok only when the script exited 0. Never raises. Exit 0 does not prove delivery (module docstring)."""
    try:
        script, enabled = _notify_settings(root)
        if not enabled:
            return False, f"not sent: notifications.telegram_enabled is false in {POLICIES_PATH}"
        p = subprocess.run([script, message], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE, timeout=timeout)
        if p.returncode != 0:
            err = (p.stderr or b"")[-300:].decode("utf-8", "replace").strip()
            return False, f"{script} exited {p.returncode}" + (f": {err}" if err else "")
        return True, f"{script} exited 0"
    except subprocess.TimeoutExpired:
        return False, f"notification script timed out after {timeout:g}s"
    except Exception as e:  # spawn error, unreadable policy file, anything: reported, never raised
        return False, f"notification script could not be run: {e!r}"


def _send(notify: Optional[Callable[[str], Any]], message: str, root: Optional[str]) -> Tuple[bool, str]:
    """notify(message) -> True / (True, detail) on success; None means send_notification. Never raises."""
    try:
        res = send_notification(message, root) if notify is None else notify(message)
    except Exception as e:
        return False, f"notifier raised {e!r}"
    if isinstance(res, tuple):
        return res[0] is True, str(res[1]) if len(res) > 1 else ""
    return res is True, "" if res is True else f"notifier returned {res!r}"


def _new_message(fid: str, model: Any, n: Any, threshold: Any, prev: Any) -> str:
    return (f"{TAG}\nNEW flag {fid}\nModel {model}: {n} critical grades in a row (limit {threshold})\n"
            + (f"Follows {prev}: its why-report is attached and the family is still critical.\n" if prev else "")
            + "Kept in the comparison (benching off). A why-report is needed.")


def _notified_state(root: Optional[str], log: logging.Logger) -> Tuple[set, set]:
    """(flag_ids with a sent NEW notification, flag_ids with a failed attempt). Old lines with no
    status count as sent."""
    recs, bad = _read_jsonl(_path(root, NOTIFIED_PATH))
    if bad:
        log.warning("%s %s: %d unreadable line(s)", TAG, NOTIFIED_PATH, len(bad))
    sent = {r.get("flag_id") for r in recs if r.get("status", "sent") == "sent"}
    failed = {r.get("flag_id") for r in recs if r.get("status") == "failed"}
    return sent, failed


def _notify_new(fid: str, message: str, log: logging.Logger, notify, root: Optional[str],
                retry: bool = False) -> bool:
    """One NEW-flag notification attempt, recorded (append-only) whatever the outcome."""
    ok, detail = _send(notify, message, root)
    rec = {"schema": NOTIFIED_SCHEMA, "flag_id": fid, "kind": "new", "status": "sent" if ok else "failed",
           "detail": detail, "retry": retry, ("notified_at" if ok else "attempted_at"): _now()}
    if ok:
        log.warning("%s NEW flag %s: notification sent (%s)", TAG, fid, detail)
    else:
        log.error("%s NEW flag %s: notification NOT sent (%s); recorded, tried again at the next health "
                  "check or run start", TAG, fid, detail)
    _append(_path(root, NOTIFIED_PATH), rec)
    return ok


def _retry_failed(log: logging.Logger, notify, root: Optional[str], skip: Optional[str] = None) -> None:
    """Try again every open flag whose NEW notification failed and was never sent."""
    sent, failed = _notified_state(root, log)
    pending = failed - sent - {skip}
    if not pending:
        return
    for fid, g in open_flags(root).items():
        if fid in pending:
            _notify_new(fid, _new_message(fid, g["model_type"], g["max_consecutive_critical"], g["threshold"],
                                          g.get("previous_flag_id")), log, notify, root, retry=True)


# ── WATCHER hooks (never raise) ─────────────────────────────────────────────

def watcher_run_start(log: logging.Logger, notify: Optional[Callable[[str], Any]] = None,
                      root: Optional[str] = None) -> List[str]:
    """At the start of every WATCHER run: open flags at the top of the summary (stdout) and the log, a
    retry of any failed NEW-flag notification, and one reminder notification for the run while any flag
    is open. Silent when nothing is open. notify None: send_notification."""
    try:
        lines = summary_lines(root)
        if not lines:
            return []
        print("\n".join(lines), flush=True)
        for l in lines:
            log.warning(l)
        _retry_failed(log, notify, root)
        opened = open_flags(root)
        if opened:
            ids = ", ".join(list(opened)[:5])
            ok, detail = _send(notify, f"{TAG}\nReminder: {len(opened)} open why-report flag(s): {ids}\n"
                                       f"Families kept in the comparison (benching off); a why-report is needed.",
                               root)
            if ok:
                log.warning("%s reminder: notification sent (%s)", TAG, detail)
            else:
                log.error("%s reminder: notification NOT sent (%s)", TAG, detail)
        return lines
    except Exception as e:  # never stops the pipeline; logged, not swallowed
        log.error("%s flag reader failed at run start (pipeline continues): %r", TAG, e, exc_info=True)
        return []


def watcher_on_health(health: Dict[str, Any], log: logging.Logger,
                      notify: Optional[Callable[[str], Any]] = None,
                      root: Optional[str] = None) -> Optional[str]:
    """Where WATCHER handles the health check result: when it carries why_report_needed, one notification
    per NEW flag_id (attempts recorded in NOTIFIED_PATH), none for a repeat of a sent one; a failed attempt
    is tried again here or at run start. Returns 'new', 'failed', 'repeat' or None."""
    try:
        flag = (health or {}).get("why_report_needed")
        fid = flag_id_of(flag) if flag else None
        _retry_failed(log, notify, root, skip=fid)
        if not flag:
            return None
        sent, _ = _notified_state(root, log)
        if fid in sent:
            log.warning("%s %s still critical (%sx, repeat of an open flag; notification already sent)",
                        TAG, fid, flag.get("consecutive_critical"))
            return "repeat"
        ok = _notify_new(fid, _new_message(fid, flag.get("model_type"), flag.get("consecutive_critical"),
                                           flag.get("threshold"), flag.get("previous_flag_id")), log, notify, root)
        return "new" if ok else "failed"
    except Exception as e:  # never stops the pipeline; logged, not swallowed
        log.error("%s flag handling failed after the health check (pipeline continues): %r",
                  TAG, e, exc_info=True)
        return None


# ── CLI ─────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Why-report flags: list open flags, attach a why-report.")
    ap.add_argument("--root", default=None, help="repo root (default: the current directory)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("list", help="open flags grouped by flag_id")
    pl.add_argument("--all", action="store_true", help="include flags that have a why-report")
    pl.add_argument("--json", action="store_true")
    pa = sub.add_parser("attach", help="attach a why-report to a flag_id (append-only)")
    pa.add_argument("flag_id")
    pa.add_argument("report_path")
    pa.add_argument("--by", required=True)
    pa.add_argument("--note", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "list":
        rf = read_flags(a.root)
        groups = [g for g in rf["groups"].values() if a.all or g["open"]]
        if a.json:
            js = {"flags": groups, "bad_lines": rf["bad_lines"], "orphan_attachments": rf["orphan_attachments"]}
            if rf.get("attempts_error"):      # [W1] only when the attempts file cannot be read
                js["attempts_error"] = rf["attempts_error"]
            print(json.dumps(js, indent=2, default=str))
        else:
            if not groups:
                print("no open why-report flags" if not a.all else "no why-report flags")
            for g in groups:
                state = "OPEN" if g["open"] else f"report {g['attachments'][0].get('report_path')}"
                if g.get("status") == STATUS_FAILED:      # [W1] only with an attempts file
                    state = f"WHY_REPORT_FAILED ({g['attempts'][-1].get('reason')}, {g['attempt_count']} attempt(s))"
                print(f"{g['flag_id']}  {g['model_type']}  {g['max_consecutive_critical']}x/{g['threshold']}  "
                      f"flagged {g['repeats']}x  last {g['last_flagged_at']}  {state}")
            for fn, nums in rf["bad_lines"].items():
                print(f"WARNING {fn}: unreadable lines {nums}", file=sys.stderr)
        if rf.get("attempts_error"):
            print(f"ERROR {rf['attempts_error']}", file=sys.stderr)
        return 0
    try:
        rec = attach(a.flag_id, a.report_path, a.by, a.root, a.note)
    except ValueError as e:
        print(f"ERROR {e}", file=sys.stderr)
        return 2
    print(json.dumps(rec, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
