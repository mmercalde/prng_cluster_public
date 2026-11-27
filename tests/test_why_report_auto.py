#!/usr/bin/env python3
"""
tests/test_why_report_auto.py - self-test for unit W2 (2026-11-27): WATCHER's automatic why-report hook
(why_report_auto.auto_run, one runner CHILD process), the flag's model identity (training_health_check: unbound on
5f6c192, never a wrong sha), the runner explaining a named model (why_report_runner, flagged_model /
checkpoint_by_sha), and the owner patch docs/drafts/unit_w/W2_OWNER_LAND.patch.

Gates
  W2.off_no_child              switch off: returns at once, no child started, nothing written, nothing logged
  W2.missing_is_off            no file / no section / no key (also the tree's own watcher_policies.json) = off
  W2.bad_value_refused         every bad value: ERROR logged, no child, None returned
  W2.empty_roots_refused       switch on with why_report_wf_roots [] or missing: ERROR, no child, no notification
  W2.on_one_child              on: exactly ONE child, argv = python3 -B <runner> --all-open --json --wf-root ...
                               --timeout-s T --root R --out-root O, in its own session; the runner is not loaded
                               into this process; stdout/stderr files under the out root; no notification when all ok
  W2.child_failure_one_alert   child dies (exit 3, SIGKILL), times out (grandchild too: whole group killed and
                               reaped), prints no JSON, prints a non-list: ONE ERROR and ONE notification each, in
                               plain words; None returned (third pass, lead decision 2)
  W2.runner_exception_contained  an Exception inside auto_run: ERROR + traceback, None; SystemExit and
                               KeyboardInterrupt are NOT caught, and the child is killed on the way out
  W2.one_notification_on_failure  failed attempts: one ERROR each, exactly one notification (also with 7 failures)
  W2.truthful_alerts           status ok + attempt_record_error: "report ATTACHED ... attempt record FAILED: <err>";
                               a failure with no reason says "reason not given"; "None" never appears (decision 3)
  W2.real_runner_wired         the real runner as the child: one failed attempt recorded (no_checkpoint), flag
                               why_report_failed, one notification, no runner .pyc written or touched
  W2.flag_identity_none        models_dir NOT given: flag lines and return values == git show 5f6c192 (3 paths)
  W2.flag_line_base_unbound    models_dir given (None, "", "  ", a relative dir holding a same-family model, an
                               absolute dir): every flag LINE == the 5f6c192 line, one line per flag (no second
                               line); the RETURNED flag has models_dir/checkpoint_sha256 None and
                               flagged_model_error "unbound: ..."; a blank dir is never read as the cwd (decisions 4-5)
  W2.compare_custom_dir_no_sha Beta R2 Q3: compare route, custom --output-dir holding an OLDER same-family model,
                               the new winner published under models/reinforcement: no sha (old or new) on any flag
  W2.stale_report_no_sha       Beta R2 Q4: critical report stamped run A, checkpoint + sidecar from run B: no sha of
                               B on any flag
  W2.runner_explains_flagged   (W1/W2 runner) a hand-written flag line naming a sha: the runner explains that model
  W2.runner_latest_line        the latest line carrying a sha decides; distinct_checkpoints
  W2.runner_flag_models_dir    the flag's own models_dir is searched last, by sha on disk
  W2.owner_patch               PRE-LANDING: the patch applies to 5f6c192 in a scratch clone; WATCHER imports there;
                               test_benching_switch and test_why_report_flags pass there; AND the tree's own owner
                               files are either exactly 5f6c192 (patch not installed) or exactly the patched
                               scratch files (same_owner True). Any third state is red (decision 6)
  W2.watcher_hook              scratch copy with the patch, run_pipeline(5, 6): normal end -> hook called once,
                               after "Pipeline execution finished"; off -> no child; in-loop safety halt, graceful
                               shutdown request, escalation (should_continue False), Step 1 clean stop -> NOT called;
                               auto_run's child failing or the module missing -> ERROR, result and steps unchanged
  W2.step5_dir_passed          the health check gets the output_dir Step 5 was given (custom, manifest default)
  W2.no_step5_none             Step 5 not run in this process: the health check gets models_dir None
  W2.step5_argv_identical      Step 5 argv with the patch == the 5f6c192 argv
  W2.installed_hook            POST-LANDING: reads the INSTALLED agents/watcher_agent.py and watcher_policies.json
                               of this tree: hook, completion guard, Step 5 dir hand-off and the three policy keys
                               present and wired (driver: normal end 1 call, halt 0, shutdown 0). Owner patch not
                               installed -> SKIP with the reason (the summary line shows it). Negative control in
                               the gate itself: the installed (or, when skipped, the patched scratch) watcher with
                               the hook removed must turn the check red (decision 6)
Mutants (a copy of one source file, the anchor found exactly once, credited when the named gate is green on the
clean file and red on the mutant; for W2.installed_hook a clean SKIP counts, because the mutant run hands the gate
the mutated installed file): see MUTANTS.

CPU only. Writes only under --out (must lie under <this tree>/data/).
  CUDA_VISIBLE_DEVICES="" taskset -c 0,1,7 nice -n 10 python3 -u -B tests/test_why_report_auto.py --out data/w2_build/t_<stamp>
"""
import argparse
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import types
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

BASE_REV = "5f6c192"   # [second pass] 689bc2b + a one-line CLAUDE.md change
PATCH = REPO / "docs" / "drafts" / "unit_w" / "W2_OWNER_LAND.patch"
OWNER_FILES = ("agents/watcher_agent.py", "watcher_policies.json")
# this unit's non-owner files, copied into the scratch clone next to the applied owner patch
NON_OWNER = ("training_health_check.py", "why_report_auto.py", "scripts/diagnostics/why_report_runner.py",
             "tests/test_why_report_flags.py", "tests/bench_switch_mutants.py", "tests/test_why_report_auto.py",
             "docs/drafts/unit_w/W2_OWNER_LAND.patch")
RESULTS = []
SKIP = "SKIP"
TS = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?\+00:00")

import why_report_flags as wrf  # noqa: E402  (the module the hook and the runner import by name)

HOOK_CALL = "                from why_report_auto import auto_run as _why_auto_run\n                _why_auto_run(logger)\n"
GUARD = '        if _why_normal_end and getattr(self, "_daemon_state", None) != "SHUTTING_DOWN":\n'

# (id, file, old, new, gate). Each `old` must occur exactly once in the file.
MUTANTS = [
    ("off_starts_child", "why_report_auto.py",
     '        if mode == "off":\n            return None\n',
     '        if mode == "off":\n            run_child(["/x"], 1, 1, root=root)\n            return None\n',
     "W2.off_no_child"),
    ("bad_value_accepted", "why_report_auto.py", "    if mode not in SWITCH_VALUES:\n", "    if False:\n",
     "W2.bad_value_refused"),
    ("exception_escapes", "why_report_auto.py", "    except Exception as e:  # never changes the pipeline's",
     "    except ZeroDivisionError as e:  # never changes the pipeline's", "W2.runner_exception_contained"),
    ("notify_per_failure", "why_report_auto.py",
     "        if attention:\n            ok, detail = send_notification(",
     "        for _each in attention:\n            ok, detail = send_notification(", "W2.one_notification_on_failure"),
    ("empty_roots_accepted", "why_report_auto.py", "    if not roots:   # [second pass", "    if False:   # [second pass",
     "W2.empty_roots_refused"),
    # [third pass] lead decision 2: process boundary (own session; the whole group killed)
    ("child_same_session", "why_report_auto.py", "                                start_new_session=True)",
     "                                start_new_session=False)", "W2.on_one_child"),
    ("child_kill_only_leader", "why_report_auto.py", "        os.killpg(proc.pid, signal.SIGKILL)\n",
     "        proc.kill()\n", "W2.child_failure_one_alert"),
    # lead decision 3: truthful alerts
    ("alert_ok_as_failed", "why_report_auto.py",
     '    if r.get("status") == "ok" and r.get("attempt_record_error"):\n        return (',
     '    if False:\n        return (', "W2.truthful_alerts"),
    # lead decision 4: a checkpoint guessed from the requested dir
    ("thc_guess_from_dir", "training_health_check.py",
     "    return {'models_dir': None, 'checkpoint_sha256': None,\n            'flagged_model_error': (f\"unbound:",
     "    _g = os.path.join(str(models_dir), 'best_model.cbm') if models_dir else ''\n"
     "    return {'models_dir': None, 'checkpoint_sha256': _sha256_file(_g) if _g and os.path.isfile(_g) else None,\n"
     "            'flagged_model_error': (f\"unbound:", "W2.compare_custom_dir_no_sha"),
    ("thc_guess_from_dir_stale", "training_health_check.py",
     "    return {'models_dir': None, 'checkpoint_sha256': None,\n            'flagged_model_error': (f\"unbound:",
     "    _g = os.path.join(str(models_dir), 'best_model.cbm') if models_dir else ''\n"
     "    return {'models_dir': None, 'checkpoint_sha256': _sha256_file(_g) if _g and os.path.isfile(_g) else None,\n"
     "            'flagged_model_error': (f\"unbound:", "W2.stale_report_no_sha"),
    # lead decision 5: the identity put on the flag line itself
    ("thc_identity_on_line", "training_health_check.py",
     "    logger.warning(\n        \"[WHY-REPORT] Model %s hit",
     "    if 'models_dir' in (run_ctx or {}):\n        flag.update(_flagged_model(model_type, run_ctx['models_dir']))\n"
     "    logger.warning(\n        \"[WHY-REPORT] Model %s hit", "W2.flag_line_base_unbound"),
    ("thc_fixed_drops_ctx", "training_health_check.py",
     "        action, consecutive, why = _severity_to_action(severity, model_type, policies, run_ctx)\n    for m, v",
     "        action, consecutive, why = _severity_to_action(severity, model_type, policies)\n    for m, v",
     "W2.flag_line_base_unbound"),
    ("runner_ignores_flag", "scripts/diagnostics/why_report_runner.py", "        if fm is None:",
     "        if True:", "W2.runner_explains_flagged"),
    ("runner_live_fallback", "scripts/diagnostics/why_report_runner.py",
     '            if ck["checkpoint_sha256"] == ck_sha:\n                return dict(ck, found_in=where)\n',
     '            if True:\n                return dict(ck, found_in=where)\n', "W2.runner_explains_flagged"),
    ("runner_first_line_decides", "scripts/diagnostics/why_report_runner.py",
     '            hit = {"checkpoint_sha256": r["checkpoint_sha256"]', '            hit = hit or {"checkpoint_sha256": r["checkpoint_sha256"]',
     "W2.runner_latest_line"),
    ("runner_no_flag_dir", "scripts/diagnostics/why_report_runner.py",
     '    for where, md in (("models_dir", models_dir), ("flag_models_dir", flag_models_dir)):',
     '    for where, md in (("models_dir", models_dir),):', "W2.runner_flag_models_dir"),
    # watcher mutants: made from the PATCHED scratch copy (W2.owner_patch) and written next to it
    ("w_hook_unguarded", "agents/watcher_agent.py", GUARD, "        if True:\n", "W2.watcher_hook"),  # decision 1
    ("w_call_drops_dir", "agents/watcher_agent.py",
     "_health = check_training_health(models_dir=self._step5_output_dir)", "_health = check_training_health()",
     "W2.step5_dir_passed"),
    ("w_init_not_none", "agents/watcher_agent.py",
     "        self._step5_output_dir = None  # [W2", '        self._step5_output_dir = "models/reinforcement"  # [W2',
     "W2.no_step5_none"),
    ("w_line_touches_argv", "agents/watcher_agent.py",
     '        if step == 5: self._step5_output_dir = final_params.get("output_dir")',
     '        if step == 5: self._step5_output_dir = final_params.get("output_dir"); cmd.append("--w2-mutant")',
     "W2.step5_argv_identical"),
    # decision 6: the INSTALLED watcher without the hook (handed to W2.installed_hook)
    ("installed_hook_removed", "INSTALLED", HOOK_CALL, "                pass\n", "W2.installed_hook"),
]


def gate(name, ok, detail=""):
    st = SKIP if ok == SKIP else bool(ok)
    RESULTS.append({"gate": name, "ok": st, "detail": str(detail)[:800]})
    print(f"[{'SKIP' if st == SKIP else ('PASS' if st else 'FAIL')}] {name} {str(detail)[:500]}", flush=True)
    return st


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_as(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1))
    return p


def tree_state(d):
    d = Path(d)
    return {str(p.relative_to(d)): sha(p) for p in sorted(d.rglob("*")) if p.is_file()}


class Cap(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records = []

    def emit(self, r):
        self.records.append((r.levelname, r.getMessage(), r.exc_info is not None))


_NLOG = [0]


def cap_logger():
    _NLOG[0] += 1
    lg = logging.getLogger(f"w2test.{_NLOG[0]}")
    lg.setLevel(logging.DEBUG)
    lg.propagate = False
    c = Cap()
    lg.addHandler(c)
    return lg, c


@contextlib.contextmanager
def patched(obj, **kw):
    saved = {k: getattr(obj, k) for k in kw}
    for k, v in kw.items():
        setattr(obj, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(obj, k, v)


class Notes:
    """Stands in for why_report_flags.send_notification (nothing is ever sent)."""
    def __init__(self, ok=True):
        self.msgs, self.ok = [], ok

    def __call__(self, message, root=None, timeout=None):
        self.msgs.append(message)
        return (self.ok, "fake exited 0" if self.ok else "fake exited 3")


def open_flag_root(d, fam="lightgbm", at="2026-11-26T01:00:00+00:00", extra=None, n=1):
    root = Path(d)
    lines = []
    for i in range(n):
        a = at if i == 0 else f"2026-11-26T0{1 + i}:30:00+00:00"
        lines.append({"schema": "ch14_why_report_flag/1", "flag_id": f"{fam}@{a}", "flagged_at": a, "model_type": fam,
                      "consecutive_critical": 3, "threshold": 3, "critical_since": a, "critical_since_known": True,
                      "benching": "off", **(extra or {})})
    (root / "diagnostics_outputs").mkdir(parents=True, exist_ok=True)
    (root / "diagnostics_outputs" / "why_report_flags.jsonl").write_text("".join(json.dumps(l) + "\n" for l in lines))
    return root, lines[0]["flag_id"]


def policy(root, td=None, raw=None):
    p = Path(root) / "watcher_policies.json"
    if raw is not None:
        p.write_text(raw)
    elif td is not None:
        p.write_text(json.dumps({"training_diagnostics": td}))
    return p


FAKE_RUNNER = r'''
import json, os, subprocess, sys, time
rec, mode, payload, code = REC, MODE, PAYLOAD, CODE
info = {"argv": sys.argv, "pid": os.getpid(), "sid": os.getsid(0), "pgid": os.getpgid(0), "ppid": os.getppid()}
if mode == "hang":
    g = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    info["grandchild"] = g.pid
with open(rec, "a") as f:
    f.write(json.dumps(info) + "\n")
if mode == "print":
    print(json.dumps(payload))
elif mode == "text":
    print(payload)
elif mode == "signal":
    os.kill(os.getpid(), 9)
elif mode == "hang":
    time.sleep(120)
sys.exit(code)
'''


def fake_runner(d, name, mode="print", payload=None, code=0):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    p, rec = d / f"fake_runner_{name}.py", d / f"fake_runner_{name}.rec.jsonl"
    p.write_text(FAKE_RUNNER.replace("REC, MODE, PAYLOAD, CODE",
                                     f"{str(rec)!r}, {mode!r}, {payload!r}, {code!r}"))
    return p, rec


def recs_of(rec):
    return [json.loads(l) for l in Path(rec).read_text().splitlines()] if Path(rec).exists() else []


def run_auto(wa, root, script, out_root=None, note=None, margin=None):
    lg, cap = cap_logger()
    note = note or Notes()
    kw = {"RUNNER_PATH": Path(script)}
    if margin is not None:
        kw["DEADLINE_MARGIN_S"] = margin
    with patched(wa, **kw), patched(sys.modules["why_report_flags"], send_notification=note):
        res = wa.auto_run(lg, root=str(root), out_root=None if out_root is None else str(out_root))
    return res, cap.records, note


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:   # a zombie counts as gone (reaped or about to be)
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return False


# ───────────────────────────────────────────────────────────────── switch gates ──
def g_off(m, d):
    root, _ = open_flag_root(d / "off")
    policy(root, {"why_report_auto": "off", "why_report_wf_roots": ["/nonexistent"], "why_report_timeout_s": 5})
    script, rec = fake_runner(d, "off")
    before = tree_state(root)
    res, recs, note = run_auto(m["wa"], root, script)
    ok = res is None and not recs_of(rec) and tree_state(root) == before and not recs and not note.msgs
    return ok, f"res {res!r} children {len(recs_of(rec))} records {recs} writes {tree_state(root) != before}"


def g_missing(m, d):
    wa, out = m["wa"], []
    cases = {"no_file": None, "empty": "{}", "no_section_key": json.dumps({"training_diagnostics": {}}),
             "null_section": json.dumps({"training_diagnostics": None}),
             "only_other_keys": json.dumps({"training_diagnostics": {"why_report_wf_roots": 5,
                                                                     "why_report_timeout_s": "x"}})}
    for name, raw in cases.items():
        root, _ = open_flag_root(d / "missing" / name)
        if raw is not None:
            policy(root, raw=raw)
        script, rec = fake_runner(d, f"missing_{name}")
        res, recs, note = run_auto(wa, root, script)
        out.append((name, res is None and not recs_of(rec) and not recs and not note.msgs, recs))
    tree = wa.read_settings(str(REPO))
    ok = all(o[1] for o in out) and tree[0] == "off" and tree[3] is None
    return ok, f"{[(o[0], o[1]) for o in out]}; tree's own policies -> {tree}"


# [second pass] the switch cases carry a good root, so only the switch check can refuse them; each ERROR names its key
_R = {"why_report_wf_roots": ["/a"]}
BAD = {"switch_ON": {"why_report_auto": "ON", **_R}, "switch_true": {"why_report_auto": True, **_R},
       "switch_1": {"why_report_auto": 1, **_R}, "switch_null": {"why_report_auto": None, **_R},
       "switch_empty": {"why_report_auto": "", **_R},
       "roots_str": {"why_report_auto": "on", "why_report_wf_roots": "/x"},
       "roots_int": {"why_report_auto": "on", "why_report_wf_roots": [1]},
       "roots_emptystr": {"why_report_auto": "on", "why_report_wf_roots": [""]},
       "timeout_0": {"why_report_auto": "on", "why_report_timeout_s": 0, **_R},
       "timeout_neg": {"why_report_auto": "on", "why_report_timeout_s": -5, **_R},
       "timeout_str": {"why_report_auto": "on", "why_report_timeout_s": "1800", **_R},
       "timeout_bool": {"why_report_auto": "on", "why_report_timeout_s": True, **_R},
       "timeout_nan": {"why_report_auto": "on", "why_report_timeout_s": float("nan"), **_R},
       "timeout_inf": {"why_report_auto": "on", "why_report_timeout_s": float("inf"), **_R}}
BAD_KEY = {"switch": "why_report_auto must be", "roots": "why_report_wf_roots must be",
           "timeout": "why_report_timeout_s must be"}
BAD_RAW = {"section_list": json.dumps({"training_diagnostics": []}), "top_list": "[]", "not_json": "{not json"}


def g_bad(m, d):
    out = []
    for name, td in list(BAD.items()) + [(k, None) for k in BAD_RAW]:
        root, _ = open_flag_root(d / "bad" / name)
        policy(root, td, raw=BAD_RAW.get(name))
        script, rec = fake_runner(d, f"bad_{name}", payload=[])
        res, recs, note = run_auto(m["wa"], root, script)
        errs = [r for r in recs if r[0] == "ERROR" and "NOT run" in r[1]]
        key = BAD_KEY.get(name.split("_")[0], "")       # the raw-file cases name no single key
        out.append((name, res is None and not recs_of(rec) and len(errs) == 1 and not note.msgs
                    and all(key in r[1] for r in errs)))
    return all(o[1] for o in out), f"{sum(o[1] for o in out)}/{len(out)} refused: {[o[0] for o in out if not o[1]]}"


def g_empty_roots(m, d):
    """[second pass, decision 4] on with no walk-forward root is a bad value."""
    out = []
    for name, td in (("roots_empty", {"why_report_auto": "on", "why_report_wf_roots": []}),
                     ("roots_missing", {"why_report_auto": "on", "why_report_timeout_s": 60})):
        root, _ = open_flag_root(d / name)
        policy(root, td)
        before = tree_state(root)
        script, rec = fake_runner(d, name, payload=[])
        res, recs, note = run_auto(m["wa"], root, script)
        errs = [r for r in recs if r[0] == "ERROR" and "NOT run" in r[1] and "why_report_wf_roots" in r[1]]
        out.append((name, res is None and not recs_of(rec) and len(errs) == 1 and not note.msgs
                    and tree_state(root) == before, [r[1][:120] for r in recs]))
    return all(o[1] for o in out), str(out)


def _runner_mods(path):
    return sum(1 for mod in list(sys.modules.values()) if getattr(mod, "__file__", None)
               and Path(mod.__file__).resolve() == Path(path).resolve())


def g_on(m, d):
    """[third pass, decision 2] one child, the documented argv, its own session, nothing loaded in-process."""
    wa = m["wa"]
    root, fid = open_flag_root(d / "on" / "root", n=2)
    policy(root, {"why_report_auto": "on", "why_report_wf_roots": ["/a", "/b"], "why_report_timeout_s": 77})
    out_root = d / "on" / "out"
    payload = [{"flag_id": fid, "status": "ok", "report_path": "/r.json"}]
    script, rec = fake_runner(d, "on", payload=payload)
    loaded_before = _runner_mods(script) + _runner_mods(wa.RUNNER_PATH)
    bc = sys.dont_write_bytecode
    res, recs, note = run_auto(wa, root, script, out_root=out_root)
    kids = recs_of(rec)
    want = [sys.executable, "-B", str(script), "--all-open", "--json", "--wf-root", "/a", "--wf-root", "/b",
            "--timeout-s", "77.0", "--root", str(root), "--out-root", str(out_root)]
    logs = sorted(p.relative_to(out_root).as_posix() for p in out_root.rglob("*") if p.is_file())
    k = kids[0] if kids else {}
    ok = (len(kids) == 1 and [sys.executable, "-B"] + k["argv"] == want and k["sid"] == k["pid"] == k["pgid"]
          and k["ppid"] == os.getpid() and res == payload and not note.msgs and sys.dont_write_bytecode == bc
          and _runner_mods(script) + _runner_mods(wa.RUNNER_PATH) == loaded_before
          and len(logs) == 2 and all(re.fullmatch(r"auto_\d{8}T\d{6}_\d+Z/runner_std(out\.json|err\.log)", x) for x in logs)
          and any(r[0] == "WARNING" and fid in r[1] for r in recs) and not any(r[0] == "ERROR" for r in recs))
    return ok, (f"children {len(kids)} argv ok {bool(kids) and [sys.executable, '-B'] + k['argv'] == want} "
                f"own session {k.get('sid') == k.get('pid')} logs {logs} notes {len(note.msgs)} records {recs[:3]}")


def g_child_failure(m, d):
    """[third pass, decision 2] a child that dies, times out or prints no usable JSON: ONE ERROR, ONE notification."""
    wa, out = m["wa"], []
    cases = (("exit3", dict(mode="exit", code=3), "died or refused"),
             ("sigkill", dict(mode="signal"), "died or refused"),
             ("no_json", dict(mode="text", payload="hello, not JSON"), "no usable JSON"),
             ("not_list", dict(mode="print", payload={"a": 1}), "no usable JSON"),
             ("hang", dict(mode="hang"), "did not finish"))
    for name, kw, words in cases:
        root, _ = open_flag_root(d / "cf" / name)
        policy(root, {"why_report_auto": "on", "why_report_wf_roots": ["/a"], "why_report_timeout_s": 1})
        script, rec = fake_runner(d, f"cf_{name}", **kw)
        t0 = time.time()
        res, recs, note = run_auto(wa, root, script, out_root=d / "cf" / f"out_{name}", margin=2)
        took = time.time() - t0
        kids = recs_of(rec)
        errs = [r for r in recs if r[0] == "ERROR"]
        gone = True
        if name == "hang":
            time.sleep(0.5)
            gone = bool(kids) and not pid_alive(kids[0]["pid"]) and not pid_alive(kids[0]["grandchild"])
        ok = (res is None and len(kids) == 1 and len(errs) == 1 and words in errs[0][1] and len(note.msgs) == 1
              and words in note.msgs[0] and "None" not in note.msgs[0] and gone and took < 30)
        out.append((name, ok, len(errs), len(note.msgs), round(took, 1), gone, (note.msgs or [""])[0][:140]))
    return all(o[1] for o in out), str(out)


def g_exc(m, d):
    """Exception: contained. SystemExit / KeyboardInterrupt: not caught, and the child is killed on the way out."""
    wa, out = m["wa"], []
    root, _ = open_flag_root(d / "exc")
    policy(root, {"why_report_auto": "on", "why_report_wf_roots": ["/a"]})
    script, rec = fake_runner(d, "exc", payload=[])

    def raiser(exc):
        def f(*a, **k):
            raise exc
        return f
    for name, exc in (("runtime", RuntimeError("synthetic runner crash")), ("oserror", OSError("synthetic"))):
        lg, cap = cap_logger()
        with patched(wa, run_child=raiser(exc)), patched(sys.modules["why_report_flags"], send_notification=Notes()):
            try:
                res, escaped = wa.auto_run(lg, root=str(root)), None
            except BaseException as e:      # must never happen
                res, escaped = "raised", repr(e)
        errs = [r for r in cap.records if r[0] == "ERROR" and r[2] and "pipeline result unchanged" in r[1]]
        out.append((name, escaped is None and res is None and len(errs) == 1))
    for name, exc in (("systemexit", SystemExit(7)), ("keyboard", KeyboardInterrupt())):
        lg, cap = cap_logger()
        with patched(wa, run_child=raiser(exc)):
            try:
                wa.auto_run(lg, root=str(root))
                got = None
            except BaseException as e:
                got = type(e)
        out.append((name, got is type(exc)))
    # KeyboardInterrupt while waiting on a real child: it propagates and the child's group is killed
    hang, hrec = fake_runner(d, "exc_hang", mode="hang")

    class InterruptingPopen(subprocess.Popen):
        def wait(self, timeout=None):
            if timeout is not None and timeout > 60:
                time.sleep(1.0)
                raise KeyboardInterrupt()
            return super().wait(timeout=timeout)
    lg, cap = cap_logger()
    sub = types.SimpleNamespace(**{k: getattr(subprocess, k) for k in dir(subprocess) if not k.startswith("__")})
    sub.Popen = InterruptingPopen
    with patched(wa, RUNNER_PATH=hang, subprocess=sub, DEADLINE_MARGIN_S=500):
        try:
            wa.auto_run(lg, root=str(root), out_root=str(d / "exc_out"))
            got = None
        except BaseException as e:
            got = type(e)
    time.sleep(0.5)
    kids = recs_of(hrec)
    out.append(("keyboard_child_killed", got is KeyboardInterrupt and len(kids) == 1 and not pid_alive(kids[0]["pid"])
                and not pid_alive(kids[0]["grandchild"])))
    return all(o[1] for o in out), str(out)


def _fails(n, prefix="f"):
    return [{"flag_id": f"{prefix}{i}", "status": "failed", "reason": "no_graded_block", "error_class": "RunnerFailure",
             "message": "m", "log_path": "/l"} for i in range(n)]


def g_notify(m, d):
    wa = m["wa"]
    root, _ = open_flag_root(d / "notify")
    policy(root, {"why_report_auto": "on", "why_report_wf_roots": ["/a"]})
    s1, _ = fake_runner(d, "n1", payload=_fails(3) + [{"flag_id": "ok1", "status": "ok"}], code=1)
    res, recs, note = run_auto(wa, root, s1, out_root=d / "n_out1")
    errs = [r for r in recs if r[0] == "ERROR" and "FAILED" in r[1]]
    a = (len(note.msgs) == 1 and all(f"f{i}: why-report FAILED (no_graded_block)" in note.msgs[0] for i in range(3))
         and "ok1" not in note.msgs[0] and len(errs) == 3)
    s2, _ = fake_runner(d, "n2", payload=_fails(7, "g"), code=1)
    res2, recs2, note2 = run_auto(wa, root, s2, out_root=d / "n_out2")
    b = len(note2.msgs) == 1 and "g4" in note2.msgs[0] and "g5" not in note2.msgs[0]
    s3, _ = fake_runner(d, "n3", payload=_fails(2), code=1)
    nn = Notes(ok=False)
    res3, recs3, _ = run_auto(wa, root, s3, out_root=d / "n_out3", note=nn)
    c = len(nn.msgs) == 1 and any(r[0] == "ERROR" and "NOT sent" in r[1] for r in recs3)
    return a and b and c, f"3 failed: notes {len(note.msgs)} errors {len(errs)}; 7 failed: notes {len(note2.msgs)}; " \
                          f"send fails: ERROR logged {c}"


def g_truthful(m, d):
    """[third pass, decision 3] Beta R2 Q8's exact shape: status ok + attempt_record_error."""
    wa = m["wa"]
    root, _ = open_flag_root(d / "truth")
    policy(root, {"why_report_auto": "on", "why_report_wf_roots": ["/a"]})
    shape = {"format": "tfm.w1.why_report_runner.result.v1", "flag_id": "F1", "status": "ok",
             "report_path": "/rep/attribution_cycle.json", "attempt_record_error": "OSError(28, 'No space left')"}
    noreason = {"flag_id": "F2", "status": "failed", "error_class": "X", "message": "m"}
    s, _ = fake_runner(d, "truth", payload=[shape, noreason], code=1)
    res, recs, note = run_auto(wa, root, s, out_root=d / "truth_out")
    msg = note.msgs[0] if note.msgs else ""
    errs = [r[1] for r in recs if r[0] == "ERROR"]
    e1 = [e for e in errs if "F1" in e]
    ok = (len(note.msgs) == 1 and "F1: report ATTACHED (/rep/attribution_cycle.json); writing the attempt record "
          "FAILED: OSError(28, 'No space left')" in msg and "F1: why-report FAILED" not in msg
          and "F2: why-report FAILED (reason not given)" in msg and "None" not in msg
          and len(e1) == 1 and "ATTACHED" in e1[0] and "attempt record FAILED" in e1[0] and "reason None" not in e1[0]
          and not any("reason None" in e for e in errs))
    return ok, f"notification {msg!r}; errors {errs}"


def g_real_runner(m, d):
    """The real runner as the child (no model dir in the root => no_checkpoint)."""
    wa = m["wa"]
    root, fid = open_flag_root(d / "real" / "root")
    policy(root, {"why_report_auto": "on", "why_report_wf_roots": [str(d / "real" / "no_wf")],
                  "why_report_timeout_s": 60})
    real = Path(wa.RUNNER_PATH)
    out_base = d / "real" / "why_out"               # under <tree>/data/, as the runner requires
    pycd = real.parent / "__pycache__"
    pyc_before = {str(p): p.stat().st_mtime_ns for p in pycd.glob("why_report_runner*.pyc")}
    loaded_before = _runner_mods(real)
    lg, cap = cap_logger()
    note = Notes()
    with patched(sys.modules["why_report_flags"], send_notification=note):
        res = wa.auto_run(lg, root=str(root), out_root=str(out_base))
    recs = cap.records
    pyc_after = {str(p): p.stat().st_mtime_ns for p in pycd.glob("why_report_runner*.pyc")}
    g = wrf.read_flags(str(root))["groups"][fid]
    ok = (isinstance(res, list) and len(res) == 1 and res[0]["status"] == "failed" and res[0]["reason"] == "no_checkpoint"
          and g["open"] and g.get("status") == "why_report_failed" and g.get("attempt_count") == 1
          and len(note.msgs) == 1 and fid in note.msgs[0] and "no_checkpoint" in note.msgs[0]
          and pyc_after == pyc_before and _runner_mods(real) == loaded_before
          and any(r[0] == "ERROR" and "no_checkpoint" in r[1] for r in recs)
          and Path(res[0]["out_dir"]).resolve().is_relative_to(out_base.resolve()))
    return ok, f"res {[(r.get('status'), r.get('reason')) for r in res or []]} flag {g.get('status')} " \
               f"attempts {g.get('attempt_count')} notes {len(note.msgs)} no runner .pyc {pyc_after == pyc_before} " \
               f"records {[r[1][:100] for r in recs if r[0] == 'ERROR']}"



# ───────────────────────────────────────────────────────────── flag line gates ──
def critical_report(mt):
    return {"model_type": mt, "status": "complete",
            "training_summary": {"rounds_captured": 17, "final_train_loss": 0.9, "final_val_loss": 0.95,
                                 "best_val_round": 14},
            "diagnosis": {"severity": "critical", "issues": ["synthetic critical"], "suggested_fixes": []}}


def fixed_report(mt, critical):
    best, vl = (1, 2.0) if critical else (14, 1.05)
    return {"model_type": mt, "status": "complete",
            "training_summary": {"best_val_round": best, "rounds_captured": 17, "final_train_loss": 1.0,
                                 "final_val_loss": vl},
            "diagnosis": {"severity": "critical" if critical else "ok", "issues": [f"{mt}-ISSUE"],
                          "suggested_fixes": [f"{mt}-FIX"]}}


PATHS = ("single", "multi", "fixed")
FLAGGED = {"single": "neural_net", "multi": "catboost", "fixed": "catboost"}


def setup_path(cwd, path):
    dd = Path(cwd) / "diagnostics_outputs"
    rules = {"max_consecutive_critical": 2}
    if path == "single":
        write_json(dd / "training_diagnostics.json", critical_report("neural_net"))
        write_json(Path(cwd) / "watcher_policies.json", {"training_diagnostics": {"model_skip_rules": rules}})
    elif path == "multi":
        write_json(dd / "training_diagnostics.json",
                   {"models": {"neural_net": critical_report("neural_net"), "catboost": critical_report("catboost")},
                    "comparison": {"winner": "catboost"}})
        write_json(Path(cwd) / "watcher_policies.json", {"training_diagnostics": {"model_skip_rules": rules}})
    else:
        write_json(dd / "catboost_diagnostics.json", fixed_report("catboost", True))
        write_json(dd / "neural_net_diagnostics.json", fixed_report("neural_net", False))
        write_json(dd / "training_diagnostics.json", fixed_report("neural_net", False))
        write_json(dd / "compare_models_summary_latest.json",
                   {"run_id": "R1", "models": {"catboost": {}, "neural_net": {}},
                    "winner_best_effort": {"model_type": "catboost", "score": 0.1}})
        write_json(Path(cwd) / "watcher_policies.json",
                   {"training_diagnostics": {"grading_rules": "fixed", "model_skip_rules": rules}})


def run_checks(thc, cwd, path, n=3, prep=None, **kw):
    cwd = Path(cwd)
    cwd.mkdir(parents=True)
    setup_path(cwd, path)
    if prep:
        prep(cwd)
    old = os.getcwd()
    os.chdir(cwd)
    try:
        res = [thc.check_training_health(**kw) for _ in range(n)]
        fp = cwd / "diagnostics_outputs" / "why_report_flags.jsonl"
        lines = [json.loads(l) for l in fp.read_text().splitlines()] if fp.exists() else []
    finally:
        os.chdir(old)
    return res, lines


def norm(x):
    return json.loads(TS.sub("<T>", json.dumps(x, sort_keys=True, default=str)))


def models_dir_for(base, fam, name="best_model.cbm", meta=None, ck_bytes=b"checkpoint bytes"):
    md = Path(base)
    md.mkdir(parents=True, exist_ok=True)
    ck = md / name
    if ck_bytes is not None:
        ck.write_bytes(ck_bytes + fam.encode())
    (md / "best_model.meta.json").write_text(json.dumps(meta if meta is not None else
                                                        {"model_type": fam, "checkpoint_path": str(ck)}))
    return md, ck


def thc_base(d):
    p = d / "base_thc" / "training_health_check.py"
    if not p.exists():
        p.parent.mkdir(parents=True)
        p.write_bytes(subprocess.run(["git", "-C", str(REPO), "show", f"{BASE_REV}:training_health_check.py"],
                                     check=True, capture_output=True).stdout)
    return load_as("thc_base_5f6c192", p)


def g_identity(m, d):
    """models_dir NOT given: every flag line and return value == 5f6c192."""
    base, out = thc_base(d), []
    for path in PATHS:
        rb, lb = run_checks(base, d / "ident" / path / "base", path)
        rn, ln = run_checks(m["thc"], d / "ident" / path / "new", path)
        same = norm(rb) == norm(rn) and norm(lb) == norm(ln)
        out.append((path, same and len(lb) == 2 and lb[0]["model_type"] == FLAGGED[path], len(lb)))
    return all(o[1] for o in out), f"(path, identical and flagged, base flag lines) {out}"


ID_KEYS = ("models_dir", "checkpoint_sha256", "flagged_model_error")


def _strip(res):
    out = []
    for r in res:
        r = copy.deepcopy(r)
        if isinstance(r.get("why_report_needed"), dict):
            for k in ID_KEYS:
                r["why_report_needed"].pop(k, None)
        out.append(r)
    return out


def _unbound(res, blank=False):
    fl = [r["why_report_needed"] for r in res if isinstance(r.get("why_report_needed"), dict)]
    return len(fl) == 2 and all(f.get("models_dir") is None and f.get("checkpoint_sha256") is None
                                and str(f.get("flagged_model_error", "")).startswith("unbound:")
                                and (not blank or "never read as the cwd" in f["flagged_model_error"]) for f in fl)


def g_line_base_unbound(m, d):
    """[third pass, decisions 4-5] the LINE is the 5f6c192 line (no second line); the RETURN carries 'unbound'."""
    base, out = thc_base(d), []
    for path in PATHS:
        fam = FLAGGED[path]
        w = d / "lbu" / path
        rb, lb = run_checks(base, w / "base", path)
        md, ck = models_dir_for(w / "md", fam)
        for name, mdv in (("none", None), ("empty", ""), ("blank", "  "), ("rel", os.path.relpath(md, w / "rel")),
                          ("abs", str(md))):
            res, ln = run_checks(m["thc"], w / name, path, models_dir=mdv)
            blob = json.dumps([res, ln], default=str)
            ok = (norm(ln) == norm(lb) and len(ln) == len(lb) == 2 and not any(k in l for l in ln for k in ID_KEYS)
                  and norm(_strip(res)) == norm(rb) and _unbound(res, blank=name in ("none", "empty", "blank"))
                  and sha(ck) not in blob)
            out.append((f"{path}/{name}", ok))
    return all(o[1] for o in out), f"{[o for o in out if not o[1]] or 'all ok'} ({len(out)} cases)"


def _compare_prep(cwd):
    """The state compare-models leaves (mpo:351-363): the NEW winner under models/reinforcement whatever
    --output-dir said, and the requested custom dir still holding an OLDER model of the same family."""
    new_md, new_ck = models_dir_for(Path(cwd) / "models" / "reinforcement", "catboost", ck_bytes=b"NEW winner ")
    old_md, old_ck = models_dir_for(Path(cwd) / "models" / "w2_custom_out", "catboost", ck_bytes=b"OLDER model ")
    return sha(new_ck), sha(old_ck)


def g_compare_custom(m, d):
    """[third pass] Beta R2 Q3 reproduced: compare route + custom dir with an older same-family model."""
    base, out = thc_base(d), []
    for path in ("multi", "fixed"):
        w = d / "cmp" / path
        rb, lb = run_checks(base, w / "base", path)
        shas = {}
        res, ln = run_checks(m["thc"], w / "new", path, models_dir="models/w2_custom_out",
                             prep=lambda c: shas.update(zip(("new", "old"), _compare_prep(c))))
        blob = json.dumps([res, ln], default=str)
        ok = (len(shas) == 2 and shas["new"] != shas["old"] and shas["new"] not in blob and shas["old"] not in blob
              and norm(ln) == norm(lb) and _unbound(res))
        out.append((path, ok, len(ln), (res[1].get("why_report_needed") or {}).get("flagged_model_error", "")[:90]))
    return all(o[1] for o in out), str(out)


def g_stale(m, d):
    """[third pass] Beta R2 Q4 reproduced: critical report stamped run A, checkpoint and sidecar from run B."""
    w = d / "stale"
    w.mkdir(parents=True)
    rep = dict(critical_report("catboost"), diagnostics_level="full", diagnostics_run_id="s5-20261101T000000Z-aaaaaaaa",
               diagnostics_run_started_at="2026-11-01T00:00:00+00:00")
    write_json(w / "diagnostics_outputs" / "training_diagnostics.json", rep)
    write_json(w / "watcher_policies.json", {"training_diagnostics": {"model_skip_rules": {"max_consecutive_critical": 2}}})
    md, ck = models_dir_for(w / "models" / "reinforcement", "catboost", ck_bytes=b"run B model ",
                            meta=None)
    meta = json.loads((md / "best_model.meta.json").read_text())
    meta["agent_metadata"] = {"run_id": "step5_20261127_090000"}
    (md / "best_model.meta.json").write_text(json.dumps(meta))
    old = os.getcwd()
    os.chdir(w)
    try:
        res = [m["thc"].check_training_health(models_dir="models/reinforcement") for _ in range(3)]
        lines = [json.loads(l) for l in (w / "diagnostics_outputs" / "why_report_flags.jsonl").read_text().splitlines()]
    finally:
        os.chdir(old)
    blob = json.dumps([res, lines], default=str)
    ok = len(lines) == 2 and sha(ck) not in blob and _unbound(res) and not any(k in l for l in lines for k in ID_KEYS)
    return ok, f"flag lines {len(lines)}; run B sha on a flag: {sha(ck) in blob}; " \
               f"{(res[1].get('why_report_needed') or {}).get('flagged_model_error', '')[:120]}"


# ─────────────────────────────────────────────────────────────── runner gate ──
_FX = {}


def fixture_model(d):
    if "surv" not in _FX:
        import test_why_report_runner as trw
        _FX["trw"] = trw
        _FX["surv"], _FX["mdir"] = trw.build_model(d / "fx")
    return _FX["trw"], _FX["surv"], _FX["mdir"]


def g_runner_flagged(m, d):
    trw, surv, mdir = fixture_model(d)
    runner = m["runner"]
    rbase = Path(runner.REPO) / "data"
    out_root = d / "rf_out" if rbase.resolve() in (d / "rf_out").resolve().parents else rbase / "w2_rf_out"
    fake = [sys.executable, "-c", trw.FAKE_LL8]
    out, detail = {}, {}
    for case in ("none_field", "flagged", "null_sha", "not_found"):
        w = d / "rf" / case
        root, wf, votes1 = trw.make_world(w, surv, mdir)
        s2 = json.loads((wf / "origin_2" / "origin_summary.json").read_text())
        sha2 = s2["frozen_families"]["s42"][trw.FAM]["checkpoint_sha256"]
        live = runner.current_checkpoint(trw.FAM, str(root))["checkpoint_sha256"]
        extra = {"flagged": {"checkpoint_sha256": sha2, "models_dir": str(root / "models" / "reinforcement")},
                 "null_sha": {"checkpoint_sha256": None, "models_dir": None, "flagged_model_error": "x"},
                 "not_found": {"checkpoint_sha256": "e" * 64, "models_dir": "/nowhere"}}.get(case)
        if extra:
            fl = root / "diagnostics_outputs" / "why_report_flags.jsonl"
            line = json.loads(fl.read_text().splitlines()[0])
            line.update(extra)
            fl.write_text(json.dumps(line) + "\n")
        res = runner.run_for_flag(trw.FID, [str(wf)], root=str(root), out_root=str(out_root / case),
                                  votes_builder=trw.copy_builder(votes1), ll8_cmd=fake, analysis=trw.FAST)
        g = wrf.read_flags(str(root))["groups"][trw.FID]
        spec = json.loads(Path(res["spec_path"]).read_text()) if res.get("spec_path") else {}
        org = Path((res.get("block") or {}).get("origin_summary") or "x").parent.name
        ck = (res.get("checkpoint") or {}).get("checkpoint_sha256")
        att = (g["attachments"] or [{}])[0]
        if case == "flagged":
            out[case] = (res["status"] == "ok" and org == "origin_2" and ck == sha2 != live
                         and res["checkpoint"].get("found_in") == "walk_forward_frozen"
                         and spec.get("expect_checkpoint_sha256") == sha2 and att.get("checkpoint_sha256") == sha2
                         and res.get("flagged_model", {}).get("checkpoint_sha256") == sha2 and not g["open"])
        elif case == "not_found":
            out[case] = (res["status"] == "failed" and res["reason"] == "no_checkpoint" and g["open"]
                         and g.get("status") == "why_report_failed" and "e" * 64 in (res.get("message") or ""))
        else:   # as before W2: the live model (origins 0/1), origin 1 the latest matching block
            out[case] = (res["status"] == "ok" and org == "origin_1" and ck == live and "flagged_model" not in res
                         and spec.get("expect_checkpoint_sha256") == live)
        detail[case] = (res["status"], res.get("reason"), org, (ck or "")[:10], sha2[:10], live[:10])
    return all(out.values()), f"{out} {detail}"


def _set_flag_lines(root, lines):
    """Rewrite the fixture's one flag line as len(lines) lines of the same flag, each updated with one dict."""
    fl = root / "diagnostics_outputs" / "why_report_flags.jsonl"
    base = json.loads(fl.read_text().splitlines()[0])
    fl.write_text("".join(json.dumps(dict(base, **x)) + "\n" for x in lines))


def g_runner_latest(m, d):
    """[second pass, decision 2] repeat lines naming different checkpoints: the LATEST line decides."""
    trw, surv, mdir = fixture_model(d)
    runner = m["runner"]
    rbase = Path(runner.REPO) / "data"
    out_root = d / "rl_out" if rbase.resolve() in (d / "rl_out").resolve().parents else rbase / "w2_rl_out"
    fake = [sys.executable, "-c", trw.FAKE_LL8]
    out, detail = {}, {}
    for case in ("live_then_frozen2", "frozen2_then_live", "same_twice"):
        w = d / "rl" / case
        root, wf, votes1 = trw.make_world(w, surv, mdir)
        s2 = json.loads((wf / "origin_2" / "origin_summary.json").read_text())
        sha2 = s2["frozen_families"]["s42"][trw.FAM]["checkpoint_sha256"]
        live = runner.current_checkpoint(trw.FAM, str(root))["checkpoint_sha256"]
        md = str(root / "models" / "reinforcement")
        order = {"live_then_frozen2": [live, sha2], "frozen2_then_live": [sha2, live], "same_twice": [sha2, sha2]}[case]
        _set_flag_lines(root, [{"checkpoint_sha256": h, "models_dir": md, "consecutive_critical": 3 + i,
                                "flagged_at": f"2026-11-26T0{1 + i}:00:00+00:00"} for i, h in enumerate(order)])
        fm = runner.flagged_model(trw.FID, str(root))
        if case == "same_twice":      # no LL-8 run needed: only the count is checked
            out[case] = fm["checkpoint_sha256"] == sha2 and fm["distinct_checkpoints"] == 1
            detail[case] = fm
            continue
        res = runner.run_for_flag(trw.FID, [str(wf)], root=str(root), out_root=str(out_root / case),
                                  votes_builder=trw.copy_builder(votes1), ll8_cmd=fake, analysis=trw.FAST)
        org = Path((res.get("block") or {}).get("origin_summary") or "x").parent.name
        ck = (res.get("checkpoint") or {}).get("checkpoint_sha256")
        want_sha, want_org = order[-1], ("origin_2" if order[-1] == sha2 else "origin_1")
        out[case] = (res["status"] == "ok" and ck == want_sha and org == want_org
                     and res.get("flagged_model", {}).get("checkpoint_sha256") == want_sha
                     and res["flagged_model"].get("distinct_checkpoints") == 2 and sha2 != live)
        detail[case] = (res["status"], res.get("reason"), org, (ck or "")[:10], res.get("flagged_model"))
    return all(out.values()), f"{out} {detail}"


def g_runner_flag_dir(m, d):
    """[second pass, decision 3] the flag's own models_dir is searched last, by sha256 hashed on disk."""
    trw, surv, mdir = fixture_model(d)
    runner = m["runner"]
    rbase = Path(runner.REPO) / "data"
    out_root = d / "fd_out" if rbase.resolve() in (d / "fd_out").resolve().parents else rbase / "w2_fd_out"
    fake = [sys.executable, "-c", trw.FAKE_LL8]
    out, detail = {}, {}
    for case in ("in_flag_dir", "flag_dir_holds_other"):
        w = d / "fd" / case
        root, wf, votes1 = trw.make_world(w, surv, mdir)
        fmd, fck = models_dir_for(w / "flag_md", trw.FAM, name="flag_model.txt", ck_bytes=b"W2 flag-dir checkpoint ")
        sha3 = sha(fck)
        want = sha3 if case == "in_flag_dir" else "f" * 64
        _set_flag_lines(root, [{"checkpoint_sha256": want, "models_dir": str(fmd)}])
        res = runner.run_for_flag(trw.FID, [str(wf)], root=str(root), out_root=str(out_root / case),
                                  votes_builder=trw.copy_builder(votes1), ll8_cmd=fake, analysis=trw.FAST)
        g = wrf.read_flags(str(root))["groups"][trw.FID]
        ck = res.get("checkpoint") or {}
        if case == "in_flag_dir":
            # found there; the fixture has no graded block for this model, so the run stops at find_block
            unit = runner.checkpoint_by_sha(trw.FAM, sha3, [str(wf)], str(root), None, str(fmd))
            try:
                runner.checkpoint_by_sha(trw.FAM, sha3, [str(wf)], str(root), None, None)
                unit_none = "found without the flag dir"
            except runner.RunnerFailure as e:
                unit_none = e.reason
            out[case] = (res["status"] == "failed" and res["reason"] == "no_graded_block"
                         and ck.get("found_in") == "flag_models_dir" and ck.get("checkpoint_sha256") == sha3
                         and Path(ck.get("checkpoint_path", "x")).resolve() == fck.resolve()
                         and unit.get("found_in") == "flag_models_dir" and unit_none == "no_checkpoint" and g["open"])
            detail[case] = (res["status"], res.get("reason"), ck.get("found_in"), unit_none)
        else:
            out[case] = (res["status"] == "failed" and res["reason"] == "no_checkpoint" and g["open"]
                         and "flag_models_dir" in (res.get("message") or ""))
            detail[case] = (res["status"], res.get("reason"), (res.get("message") or "")[-200:])
    return all(out.values()), f"{out} {detail}"


# ─────────────────────────────────────────────────────── owner patch + WATCHER ──
def git(*a, cwd=REPO, check=False):
    return subprocess.run(["git", *a], cwd=str(cwd), capture_output=True, text=True, check=check)


_SCRATCH = []          # the scratch clone W2.owner_patch made (W2.watcher_hook runs in it)


def scratch_clone(d):
    sc = d / "scratch"
    if sc.exists():
        return sc
    git("clone", "-q", "--shared", "--no-checkout", str(REPO), str(sc), check=True)
    git("checkout", "-q", "--detach", BASE_REV, cwd=sc, check=True)
    return sc


def pytest_in(sc, test, bt, env_extra=None):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", **(env_extra or {}))
    env.pop("BENCH_CODE_ROOT", None)
    env.pop("WHY_CODE_ROOT", None)
    env.pop("BENCH_RUN_LOG", None)
    p = subprocess.run([sys.executable, "-B", "-m", "pytest", "-q", test, f"--basetemp={bt}", "-p", "no:cacheprovider"],
                       cwd=str(sc), env=env, capture_output=True, text=True, timeout=1800)
    last = [l for l in p.stdout.splitlines() if l.strip()][-1:] or [""]
    return p.returncode, last[0], p.stdout[-3000:] + p.stderr[-2000:]


def owner_state():
    """'base' (the tree's owner files == 5f6c192), 'patched' (== the archived patch applied to 5f6c192, checked by
    W2.owner_patch) or 'other'."""
    base = all(git("diff", "--quiet", BASE_REV, "--", f).returncode == 0 for f in OWNER_FILES)
    return "base" if base else "changed"


def g_owner_patch(m, d):
    """PRE-LANDING gate (third pass, decision 6): replay of the archived patch on 5f6c192, and same_owner asserted."""
    sc = scratch_clone(d)
    chk = git("apply", "--check", str(PATCH), cwd=sc)
    if chk.returncode != 0:
        return False, f"git apply --check failed on {BASE_REV}: {chk.stderr[:400]}"
    git("apply", str(PATCH), cwd=sc, check=True)
    _SCRATCH.append(sc)
    changed = sorted(git("diff", "--name-only", cwd=sc).stdout.split())
    for f in NON_OWNER:
        (sc / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / f, sc / f)
    # same_owner shows that the tree under test holds EITHER the base owner files (patch not installed) OR exactly
    # the archived patch's result; anything else means the archive and the tree disagree
    st = owner_state()
    same_owner = all(sha(sc / f) == sha(REPO / f) for f in OWNER_FILES)
    owner_ok = st == "base" or same_owner
    imp = subprocess.run([sys.executable, "-B", "-c",
                          "import sys, importlib.util as u; sys.path[:0] = [sys.argv[1]]; "
                          "s = u.spec_from_file_location('wa_scratch', sys.argv[1] + '/agents/watcher_agent.py'); "
                          "m = u.module_from_spec(s); s.loader.exec_module(m); "
                          "import why_report_auto as a; print('IMPORT_OK', hasattr(m, 'WatcherAgent'), "
                          "a.read_settings(sys.argv[1]))", str(sc)],
                         cwd=str(sc), env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"),
                         capture_output=True, text=True, timeout=600)
    imp_ok = imp.returncode == 0 and "IMPORT_OK True ('off', None, None, None)" in imp.stdout
    (sc / "data").mkdir(exist_ok=True)
    rb, lb, tb = pytest_in(sc, "tests/test_benching_switch.py", sc / "data" / "pt_bench")
    rf, lf, tf = pytest_in(sc, "tests/test_why_report_flags.py", sc / "data" / "pt_flags")
    (d / "owner_patch_suites.log").write_text(f"bench rc {rb}\n{tb}\n\nflags rc {rf}\n{tf}\n\nimport\n{imp.stdout}{imp.stderr}")
    ok = (changed == sorted(OWNER_FILES) and imp_ok and rb == 0 and rf == 0 and " passed" in lb and " passed" in lf
          and "failed" not in lb + lf and owner_ok)
    return ok, (f"apply --check rc 0; files {changed}; tree owner files: {st}, same as the patched scratch: "
                f"{same_owner} -> {'ok' if owner_ok else 'MISMATCH'}; WATCHER import ok {imp_ok}; "
                f"test_benching_switch: rc {rb} '{lb}'; test_why_report_flags: rc {rf} '{lf}'")


DRIVER = r'''
import contextlib, io, json, logging, os, sys, types
sc, root, mode, wpath = sys.argv[1:5]
sys.path[:0] = [sc]
import importlib.util as u
import why_report_auto as wa
calls = []
def run_child(roots, timeout_s, n_open, root=None, out_root=None):
    calls.append([roots, timeout_s, n_open, root])
    if mode == "raise":
        raise RuntimeError("synthetic runner crash")
    return [{"flag_id": "x", "status": "ok", "report_path": "/r"}], None, {"returncode": 0, "stderr": "/e", "stdout": "/o"}
wa.run_child = run_child
s = u.spec_from_file_location("wa_drv", wpath)
W = u.module_from_spec(s); sys.modules["wa_drv"] = W; s.loader.exec_module(W)
if mode == "hook_missing":
    sys.modules["why_report_auto"] = None          # `from why_report_auto import ...` now fails
cfg = W.WatcherConfig(use_llm=False, manifests_dir=os.path.join(sc, "agent_manifests"),
                      history_file=os.path.join(root, "h.json"), log_file=os.path.join(root, "d.jsonl"),
                      pid_file=os.path.join(root, "p.pid"), daemon_state_file=os.path.join(root, "s.json"),
                      pending_approval_file=os.path.join(root, "pa.json"), halt_file=os.path.join(root, "halt"))
agent = W.WatcherAgent(cfg)
agent.llm_lifecycle = agent.llm_router = None
steps = []
def run_step(step, params=None, **kw):
    steps.append(step)
    if mode == "shutdown" and step == 6:       # what WATCHER's SIGTERM/SIGINT handler does (_graceful_shutdown)
        agent._daemon_state = "SHUTTING_DOWN"; agent.running = False; agent._pipeline_running = False
    if mode == "clean_stop" and step == 5:
        return {"success": False, W.STEP1_SEARCH_RESULT_KEY: sorted(W.step1_search_clean_stop_values())[0]}
    return {"success": True, "confidence": 0.9}
agent.run_step = run_step
agent.evaluate_results = lambda step, results, n: (types.SimpleNamespace(recommended_action="proceed", confidence=0.9), {})
def execute(decision, context):
    if mode == "escalate":
        return False
    agent.current_step += 1; return True
agent.execute_decision = execute
agent._invalidate_step_freshness = lambda step: None
health = {"action": "PROCEED", "model_type": "catboost", "severity": "ok", "issues": [], "suggested_fixes": [],
          "confidence": 0.9}
safety = []
def check_safety():
    safety.append(1)
    return not (mode == "halt" and len(safety) == 3)     # start ok, step 5 ok, refused before step 6
import time as _t
ft = types.SimpleNamespace(**{k: getattr(_t, k) for k in dir(_t) if not k.startswith("_")}); ft.sleep = lambda s: None
W.__dict__.update(check_safety=check_safety, notify_telegram=lambda m, *a, **k: None, PROGRESS_DISPLAY_AVAILABLE=False,
                  TRAINING_HEALTH_CHECK_AVAILABLE=True, check_training_health=lambda *a, **k: dict(health),
                  reset_skip_registry=lambda m: None, time=ft)
recs = []
class Cap(logging.Handler):
    def emit(self, r):
        recs.append([r.levelname, r.getMessage(), r.exc_info is not None, len(calls)])
logging.getLogger("WatcherAgent").addHandler(Cap())
os.chdir(root)
with contextlib.redirect_stdout(io.StringIO()):
    ret = agent.run_pipeline(start_step=5, end_step=6)
print("RESULT " + json.dumps({"returned": ret, "steps": steps, "calls": calls, "records": recs}, default=str))
'''
HOOK_MODES = ("off", "on", "raise", "hook_missing", "halt", "shutdown", "escalate", "clean_stop")
HOOK_STEPS = {"halt": [5], "escalate": [5], "clean_stop": [5]}


def run_driver(d, sc, wpath, mode, tag):
    drv = d / "watcher_driver.py"
    if not drv.exists():
        drv.write_text(DRIVER)
    root = d / "wh" / tag / mode
    root.mkdir(parents=True)
    open_flag_root(root)                                     # one open flag, so the hook has work
    if mode != "off":
        policy(root, {"why_report_auto": "on", "why_report_wf_roots": ["/wf"], "why_report_timeout_s": 9})
    p = subprocess.run([sys.executable, "-B", str(drv), str(sc), str(root), mode, str(wpath)], cwd=str(sc),
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"),
                       capture_output=True, text=True, timeout=600)
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    if p.returncode != 0 or not line:
        raise RuntimeError(f"{mode}: driver rc {p.returncode}: {p.stderr[-800:]}")
    return json.loads(line[0][7:])


def hook_behaviour(d, sc, wpath, tag, modes=HOOK_MODES):
    """(ok, detail): normal end -> one call after 'finished'; off/halt/shutdown/escalate/clean stop -> none."""
    out = {mode: run_driver(d, sc, wpath, mode, tag) for mode in modes}
    want_calls = {"on": 1, "raise": 1}
    bad = []
    for mode, r in out.items():
        n = len(r["calls"])
        if n != want_calls.get(mode, 0):
            bad.append(f"{mode}: {n} calls")
        if r["steps"] != HOOK_STEPS.get(mode, [5, 6]):
            bad.append(f"{mode}: steps {r['steps']}")
        if r["returned"] is not None:
            bad.append(f"{mode}: returned {r['returned']!r}")
    if "on" in out:
        fin = [i for i, r in enumerate(out["on"]["records"]) if r[1] == "Pipeline execution finished"]
        if not fin or any(r[3] for r in out["on"]["records"][:fin[0] + 1]):
            bad.append("on: hook not after 'finished'")
        if out["on"]["calls"] and out["on"]["calls"][0][:3] != [["/wf"], 9, 1]:
            bad.append(f"on: call {out['on']['calls']}")
    for mode, words in (("raise", "pipeline result unchanged"), ("hook_missing", "hook failed")):
        if mode in out:
            errs = [r for r in out[mode]["records"] if r[0] == "ERROR" and r[2]]
            if len(errs) != 1 or words not in errs[0][1]:
                bad.append(f"{mode}: errors {errs}")
    for mode in set(out) - {"raise", "hook_missing"}:   # halt: only WATCHER's own base line
        if any(r[0] == "ERROR" and not (mode == "halt" and r[1] == "Safety halt detected - stopping pipeline")
               for r in out[mode]["records"]):
            bad.append(f"{mode}: unexpected ERROR")
    if "halt" in out and not any(r[1] == "Safety halt detected - stopping pipeline" for r in out["halt"]["records"]):
        bad.append("halt: the in-loop safety halt did not happen")
    calls = {k: len(v["calls"]) for k, v in out.items()}
    steps = {k: v["steps"] for k, v in out.items()}
    return not bad, f"calls {calls}; steps {steps}; problems {bad}"


def g_watcher_hook(m, d):
    """[third pass, decision 1] the hook runs only after a normal end."""
    sc = _SCRATCH[-1] if _SCRATCH else d / "missing"
    if not (sc / "why_report_auto.py").exists():
        return False, "scratch copy missing (W2.owner_patch did not run)"
    wpath = Path(m.get("watcher") or sc / "agents" / "watcher_agent.py")
    return hook_behaviour(d, sc, wpath, "scratch")


INSTALLED_MARKERS = {
    "init": '        self._step5_output_dir = None  # [W2',
    "step5_handoff": '        if step == 5: self._step5_output_dir = final_params.get("output_dir")',
    "health_dir": "_health = check_training_health(models_dir=self._step5_output_dir)",
    "guard_init": "        _why_normal_end = False  # [W2",
    "guard_set": "                _why_normal_end = self._pipeline_running and self.current_step > end_step",
    "guard": GUARD.rstrip("\n"),
    "hook": "from why_report_auto import auto_run as _why_auto_run",
}
POLICY_KEYS = ("why_report_auto", "why_report_wf_roots", "why_report_timeout_s")


def check_installed(d, sc, wpath, ppath, tag):
    """(status, detail) for an installed watcher + policies: 'not_installed', True or False."""
    src = Path(wpath).read_text()
    have = {k: src.count(v) for k, v in INSTALLED_MARKERS.items()}
    try:
        td = (json.loads(Path(ppath).read_text()).get("training_diagnostics") or {})
    except Exception as e:
        td, perr = {}, repr(e)
    else:
        perr = None
    keys = [k for k in POLICY_KEYS if k in td]
    if not any(have.values()) and not keys and perr is None:
        return "not_installed", "owner patch not installed (no W2 line in agents/watcher_agent.py, no W2 policy key)"
    bad = [f"{k} x{n}" for k, n in have.items() if n != 1] + [f"policy key {k} missing" for k in POLICY_KEYS if k not in td]
    if perr:
        bad.append(f"policies unreadable {perr}")
    if not bad:
        pol_dir = d / "inst_pol" / tag
        pol_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ppath, pol_dir / "watcher_policies.json")
        sys.path.insert(0, str(sc))
        try:
            wa_sc = load_as(f"wra_inst_{tag}", sc / "why_report_auto.py")
        finally:
            sys.path.remove(str(sc))
        st = wa_sc.read_settings(str(pol_dir))
        if st[3] is not None or st[0] != td.get("why_report_auto"):
            bad.append(f"policy values not valid: {st}")
    if bad:
        return False, f"installed but not wired: {bad}"
    run_path = Path(sc) / "agents" / f"watcher_agent_chk_{tag}.py"     # next to the scratch: REPO_ROOT is the scratch
    if Path(wpath).resolve() != run_path.resolve():
        shutil.copy2(wpath, run_path)
    ok, det = hook_behaviour(d, sc, run_path, f"inst_{tag}", modes=("on", "halt", "shutdown"))
    return ok, f"markers ok, policy keys ok; behaviour: {det}"


def g_installed(m, d):
    """POST-LANDING gate (third pass, decision 6): the INSTALLED owner files of the tree under test."""
    sc = _SCRATCH[-1] if _SCRATCH else None
    if sc is None:
        return False, "scratch copy missing (W2.owner_patch did not run)"
    wpath = Path(m.get("installed") or REPO / "agents" / "watcher_agent.py")
    ppath = REPO / "watcher_policies.json"
    if m.get("installed"):                                    # mutant run: the mutated installed file's policies
        ppath = Path(m["installed_policies"])
    st, det = check_installed(d, sc, wpath, ppath, "tree")
    # negative control: the same check on a copy with the hook call removed must be red
    ctl_src = wpath if st != "not_installed" else sc / "agents" / "watcher_agent.py"
    ctl_pol = ppath if st != "not_installed" else sc / "watcher_policies.json"
    ctl = d / "control" / "agents"
    ctl.mkdir(parents=True, exist_ok=True)
    text = Path(ctl_src).read_text()
    ctl_path = sc / "agents" / "watcher_agent_ctl_no_hook.py"     # next to the scratch file: REPO_ROOT is the scratch
    ctl_path.write_text(text.replace(HOOK_CALL, "                pass\n"))
    c_st, c_det = check_installed(d, sc, ctl_path, ctl_pol, "control")
    control_red = text.count(HOOK_CALL) == 1 and c_st is False
    if st == "not_installed":
        ref_st, ref_det = check_installed(d, sc, sc / "agents" / "watcher_agent.py", sc / "watcher_policies.json", "ref")
        self_ok = ref_st is True and control_red
        return (SKIP if self_ok else False), (f"SKIP: {det}. Gate self-check on the patched scratch copy: {ref_st}; "
                                              f"negative control (hook removed) red: {control_red}")
    return (st is True and control_red), f"installed: {st}: {det}; negative control (hook removed) red: {control_red} ({c_det[:160]})"


DRIVER5 = r'''
import contextlib, io, json, logging, os, sys, types
sc, root, wpath, mode, custom = sys.argv[1:6]
sys.path[:0] = [sc]
import importlib.util as u
s = u.spec_from_file_location("wa_drv5", wpath)
W = u.module_from_spec(s); sys.modules["wa_drv5"] = W; s.loader.exec_module(W)
cfg = W.WatcherConfig(use_llm=False, manifests_dir=os.path.join(sc, "agent_manifests"),
                      history_file=os.path.join(root, "h.json"), log_file=os.path.join(root, "d.jsonl"),
                      pid_file=os.path.join(root, "p.pid"), daemon_state_file=os.path.join(root, "s.json"),
                      pending_approval_file=os.path.join(root, "pa.json"), halt_file=os.path.join(root, "halt"))
agent = W.WatcherAgent(cfg)
agent.llm_lifecycle = agent.llm_router = None
init_attr = getattr(agent, "_step5_output_dir", "<absent>")
argvs = []
class Done:
    returncode, stdout, stderr = 0, "", ""
def stream(cmd, step, timeout):            # the dispatch boundary (no child is started)
    argvs.append([step, list(cmd)]); return Done()
# stub boundary as tests/test_registry_ll_register.py _build_argv; the command builder is not stubbed
agent._ensure_execution_set = lambda p=None: object()
agent._run_preflight_check = lambda s: (True, "W2-DRV5")
agent._run_step_streaming = stream
agent._run_post_step_cleanup = lambda *a, **k: None
agent._find_results = lambda *a, **k: {}
W.check_output_freshness = lambda s: (False, "W2-DRV5: stale", False)
W.p05_freeze_dataset = lambda run_label="watcher": types.SimpleNamespace(
    path="daily3.json", sha256="0" * 64, size=0, record_count=0, manifest_id="W2-DRV5")
W.p05_resolve_dataset_path = lambda p: str(p)
import miner.dataset_authority as _ds
for n, v in (("_active_execution_set", lambda: None), ("load_provisioning_nodes", lambda *a, **k: []),
             ("fleet_preflight", lambda *a, **k: []), ("resolve_absent_fleet_status", lambda *a, **k: "NOT_APPLICABLE"),
             ("default_provisioning_manifest_path", lambda *a, **k: "<W2-DRV5>"),
             ("write_run_provenance", lambda *a, **k: None)):
    setattr(_ds, n, v)
if mode == "no_step5":                      # Step 5 did not run in this process
    def run_step(step, params=None, **kw):
        argvs.append([step, None]); return {"success": True, "confidence": 0.9}
    agent.run_step = run_step
agent.evaluate_results = lambda step, results, n: (types.SimpleNamespace(recommended_action="proceed", confidence=0.9), {})
def execute(decision, context):
    agent.current_step += 1; return True
agent.execute_decision = execute
agent._invalidate_step_freshness = lambda step: None
health = {"action": "PROCEED", "model_type": "catboost", "severity": "ok", "issues": [], "suggested_fixes": [],
          "confidence": 0.9}
health_calls = []
def chk(*a, **k):
    health_calls.append([list(a), dict(k)]); return dict(health)
import time as _t
ft = types.SimpleNamespace(**{k: getattr(_t, k) for k in dir(_t) if not k.startswith("_")}); ft.sleep = lambda s: None
W.__dict__.update(check_safety=lambda: True, notify_telegram=lambda m, *a, **k: None, PROGRESS_DISPLAY_AVAILABLE=False,
                  TRAINING_HEALTH_CHECK_AVAILABLE=True, check_training_health=chk,
                  reset_skip_registry=lambda m: None, time=ft)
params = None if custom == "-" else {"output_dir": custom}
os.chdir(root)
with contextlib.redirect_stdout(io.StringIO()):
    ret = agent.run_pipeline(start_step=5, end_step=5, params=params)
print("RESULT " + json.dumps({"returned": ret, "argvs": argvs, "health": health_calls, "init_attr": init_attr,
                              "final_attr": getattr(agent, "_step5_output_dir", "<absent>")}, default=str))
'''
CUSTOM_OUT = "models/w2_custom_out"      # relative, as the manifest's own value is


def run_drv5(d, wpath, mode, custom, tag):
    sc = _SCRATCH[-1]
    drv = d / "watcher_driver5.py"
    if not drv.exists():
        drv.write_text(DRIVER5)
    root = d / "w5" / tag
    root.mkdir(parents=True)
    p = subprocess.run([sys.executable, "-B", str(drv), str(sc), str(root), str(wpath), mode, custom], cwd=str(sc),
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"),
                       capture_output=True, text=True, timeout=600)
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    if p.returncode != 0 or not line:
        raise RuntimeError(f"{tag}: driver rc {p.returncode}: {p.stderr[-800:]}")
    return json.loads(line[0][7:])


def _patched_watcher(m):
    if not _SCRATCH:
        return None
    return Path(m.get("watcher") or _SCRATCH[-1] / "agents" / "watcher_agent.py")


def _step5_argv(r):
    a = [x[1] for x in r["argvs"] if x[0] == 5]
    return a[0] if len(a) == 1 else None


def g_step5_dir(m, d):
    wp = _patched_watcher(m)
    if wp is None:
        return False, "scratch copy missing (W2.owner_patch did not run)"
    default = json.loads((_SCRATCH[-1] / "agent_manifests" / "reinforcement.json").read_text())["default_params"][
        "output_dir"]
    a = run_drv5(d, wp, "step5", CUSTOM_OUT, "custom")
    b = run_drv5(d, wp, "step5", "-", "default")
    av, bv = _step5_argv(a), _step5_argv(b)
    ok = (a["health"] == [[[], {"models_dir": CUSTOM_OUT}]] and b["health"] == [[[], {"models_dir": default}]]
          and av and bv and av[av.index("--output-dir") + 1] == CUSTOM_OUT and bv[bv.index("--output-dir") + 1] == default
          and a["init_attr"] is None and a["final_attr"] == CUSTOM_OUT)
    return ok, f"custom: health {a['health']} attr {a['init_attr']!r}->{a['final_attr']!r}; default: health {b['health']}"


def g_no_step5(m, d):
    wp = _patched_watcher(m)
    if wp is None:
        return False, "scratch copy missing (W2.owner_patch did not run)"
    a = run_drv5(d, wp, "no_step5", "-", "plain")
    b = run_drv5(d, wp, "no_step5", CUSTOM_OUT, "params_custom")    # params name a dir, Step 5 still not run
    ok = all(r["health"] == [[[], {"models_dir": None}]] and r["init_attr"] is None and r["final_attr"] is None
             and r["argvs"] == [[5, None]] for r in (a, b))
    return ok, f"health {a['health']} / {b['health']}; attr {a['init_attr']!r}/{b['final_attr']!r}"


def g_step5_argv(m, d):
    wp = _patched_watcher(m)
    if wp is None:
        return False, "scratch copy missing (W2.owner_patch did not run)"
    basep = _SCRATCH[-1] / "agents" / "watcher_agent_base_5f6c192.py"
    if not basep.exists():
        basep.write_bytes(subprocess.run(["git", "-C", str(REPO), "show", f"{BASE_REV}:agents/watcher_agent.py"],
                                         check=True, capture_output=True).stdout)
    out = []
    for tag, custom in (("custom", CUSTOM_OUT), ("default", "-")):
        n = run_drv5(d, wp, "step5", custom, f"new_{tag}")
        b = run_drv5(d, basep, "step5", custom, f"base_{tag}")
        nv, bv = _step5_argv(n), _step5_argv(b)
        out.append((tag, bool(nv) and nv == bv and b["health"] == [[[], {}]] and b["init_attr"] == "<absent>",
                    len(nv or []), sorted(set(nv or []) ^ set(bv or []))))
    return all(o[1] for o in out), f"(case, argv identical to {BASE_REV}, argv length, differing items) {out}"


GATES = [("W2.off_no_child", g_off), ("W2.missing_is_off", g_missing), ("W2.bad_value_refused", g_bad),
         ("W2.empty_roots_refused", g_empty_roots), ("W2.on_one_child", g_on),
         ("W2.child_failure_one_alert", g_child_failure), ("W2.runner_exception_contained", g_exc),
         ("W2.one_notification_on_failure", g_notify), ("W2.truthful_alerts", g_truthful),
         ("W2.real_runner_wired", g_real_runner),
         ("W2.flag_identity_none", g_identity), ("W2.flag_line_base_unbound", g_line_base_unbound),
         ("W2.compare_custom_dir_no_sha", g_compare_custom), ("W2.stale_report_no_sha", g_stale),
         ("W2.runner_explains_flagged", g_runner_flagged), ("W2.runner_latest_line", g_runner_latest),
         ("W2.runner_flag_models_dir", g_runner_flag_dir),
         ("W2.owner_patch", g_owner_patch), ("W2.watcher_hook", g_watcher_hook),
         ("W2.step5_dir_passed", g_step5_dir), ("W2.no_step5_none", g_no_step5), ("W2.step5_argv_identical", g_step5_argv),
         ("W2.installed_hook", g_installed)]


def modules(code_dir=REPO, tag="clean"):
    code_dir = Path(code_dir)
    return {"wa": load_as(f"why_report_auto_{tag}", code_dir / "why_report_auto.py"),
            "thc": load_as(f"thc_{tag}", code_dir / "training_health_check.py"),
            "runner": load_as(f"runner_{tag}", code_dir / "scripts" / "diagnostics" / "why_report_runner.py")}


def run_gate(fn, mods, d):
    d.mkdir(parents=True, exist_ok=True)
    try:
        return fn(mods, d)
    except Exception as e:
        import traceback
        return False, f"gate raised {e!r}: {traceback.format_exc()[-600:]}"


def run_mutants(out, clean_ok):
    res = []
    for mid, rel, old, new, gname in MUTANTS:
        md = out / "mut" / mid
        for f in ("why_report_auto.py", "training_health_check.py", "scripts/diagnostics/why_report_runner.py"):
            (md / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / f, md / f)
        extra = {}
        if rel in ("agents/watcher_agent.py", "INSTALLED"):
            if not _SCRATCH:
                print(f"[FAIL] mutant {mid}: no scratch copy", flush=True)
                res.append({"mutant": mid, "ok": False, "why": "no scratch copy"})
                continue
            # next to the patched file, so its REPO_ROOT (from __file__) is the scratch copy's
            mpath = _SCRATCH[-1] / "agents" / f"watcher_agent_mut_{mid}.py"
            src_dir = REPO if (rel == "INSTALLED" and owner_state() != "base") else _SCRATCH[-1]
            shutil.copy2(src_dir / "agents" / "watcher_agent.py", mpath)
            if rel == "INSTALLED":
                extra = {"installed": mpath, "installed_policies": src_dir / "watcher_policies.json"}
            else:
                extra = {"watcher": mpath}
        else:
            mpath = md / rel
        src = mpath.read_text()
        n = src.count(old)
        if n != 1:
            print(f"[FAIL] mutant {mid}: anchor found {n} times (must be 1)", flush=True)
            res.append({"mutant": mid, "ok": False, "why": f"anchor x{n}"})
            continue
        mpath.write_text(src.replace(old, new))
        saved_path = list(sys.path)       # a runner copy puts its own dir on sys.path: taken off again below
        try:
            mods = modules(REPO, f"m_{mid}")
            mods.update(extra)
            if not extra:
                key = {"why_report_auto.py": "wa", "training_health_check.py": "thc",
                       "scripts/diagnostics/why_report_runner.py": "runner"}[rel]
                mods[key] = load_as(f"{key}_mut_{mid}", mpath)
            fn = dict(GATES)[gname]
            ok, detail = run_gate(fn, mods, md / "work")
        finally:
            sys.path[:] = saved_path
        clean = clean_ok.get(gname)
        # W2.installed_hook: a clean SKIP counts (the mutant run hands the gate the mutated installed file)
        killed = (clean is True or (gname == "W2.installed_hook" and clean == SKIP)) and ok is False
        print(f"[{'KILLED' if killed else 'SURVIVED'}] mutant {mid} -> {gname}: {str(detail)[:300]}", flush=True)
        res.append({"mutant": mid, "gate": gname, "ok": killed, "detail": str(detail)[:600]})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", help="gate names")
    ap.add_argument("--no-mutants", action="store_true")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if not out.is_relative_to((REPO / "data").resolve()):
        sys.exit(f"--out must lie under {REPO / 'data'}")
    if out.exists():
        sys.exit(f"--out {out} exists; give a new directory")
    out.mkdir(parents=True)
    logging.getLogger().addHandler(logging.NullHandler())   # the modules' own warnings are not printed
    t0 = time.time()
    mods = modules()
    clean = {}
    for name, fn in GATES:
        if a.only and name not in a.only:
            continue
        ok, detail = run_gate(fn, mods, out / "gates" / name)
        clean[name] = gate(name, ok, detail)
    mut = [] if a.no_mutants else run_mutants(out, clean)
    n_ok = sum(r["ok"] is True for r in RESULTS)
    skipped = [r["gate"] for r in RESULTS if r["ok"] == SKIP]
    n_run = len(RESULTS) - len(skipped)
    n_m = sum(r["ok"] for r in mut)
    (out / "results.json").write_text(json.dumps({"gates": RESULTS, "mutants": mut}, indent=2))
    sk = f", {len(skipped)} SKIPPED ({', '.join(skipped)})" if skipped else ", 0 skipped"
    print(f"gates {n_ok}/{n_run} passed{sk}; mutants {n_m}/{len(mut)}; {time.time() - t0:.1f} s", flush=True)
    green = n_ok == n_run and n_m == len(mut) and (a.no_mutants or len(mut) >= 6)
    print(f"COMPLETION SENTINEL: {'PASS' if green else 'FAIL'} ({n_ok}/{n_run} gates{sk}, {n_m}/{len(mut)} "
          f"mutants killed)", flush=True)
    sys.exit(0 if green else 1)


if __name__ == "__main__":
    main()
