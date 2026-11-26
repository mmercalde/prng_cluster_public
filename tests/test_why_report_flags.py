#!/usr/bin/env python3
"""
Why-report flag reader and WATCHER look-out tests (2026-11-26; owner rulings 07:10, record
logs/e2e_session/OWNER_RULING_benching_switch_20261126.txt). CPU only, synthetic fixtures, no network:
no notification is ever sent. WATCHER's notify_telegram is replaced by a list, and why_report_flags' own
sender (send_notification, R1 F2) runs against a fake subprocess module installed for EVERY test by the
autouse fixture _no_real_notify; the real sender code runs, only its subprocess boundary is stubbed.

  R  why_report_flags.py: grouping by flag_id, open until attached, append-only attach, CLI, git-ignore
  N  R1 F2 (Beta Q6): the sender's outcomes (exit 0, spawn error, non-zero exit, timeout), the policy's
     script path, and a failed NEW-flag notification logged at ERROR, recorded as failed and tried again
  F  R1 F1 (Beta Q5 last paragraph) and F3: flag_id_of agrees with the writer for a line without
     flag_id; after an attached why-report three more criticals raise a NEW linked flag
  W  agents/watcher_agent.py hook: open flags at the top of the run's summary and log, one reminder per
     run, one notification per NEW flag_id, a reader failure logged and the pipeline goes on, and with no
     flags run_pipeline(5, 6) is identical to the WATCHER at BASE_REV (run_step calls = argv inputs,
     notifications, stdout, log)
Credited assertions carry [CREDIT:<tag>]; tests/why_flags_mutants.py turns each tag red.

Run (from the worktree root), one suite at a time:
    PYTHONDONTWRITEBYTECODE=1 nice -n 15 python3 -B -m pytest -q tests/test_why_report_flags.py --basetemp=data/why_flags/pt -p no:cacheprovider
Code under test is imported from $WHY_CODE_ROOT (default: the worktree root).
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import contextlib
import copy
import importlib.util
import inspect
import io
import json
import logging
import subprocess
import sys
import time as _time
import types
from pathlib import Path

import pytest

WORKTREE = Path(__file__).resolve().parents[1]
CODE_ROOT = Path(os.environ.get("WHY_CODE_ROOT", str(WORKTREE))).resolve()
BASE_REV = "7f28109"   # main the package is rebased on (agents/watcher_agent.py identical to 933a4a0)
sys.path.insert(0, str(WORKTREE))
sys.path.insert(0, str(CODE_ROOT))


def credit(tag):
    return f"[CREDIT:{tag}]"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


wrf = _load("why_report_flags", CODE_ROOT / "why_report_flags.py")   # the name WATCHER imports
thc = _load("thc_for_why_flags", CODE_ROOT / "training_health_check.py")
TAG = "[WATCHER][WHY-REPORT]"


class FakeSubprocess:
    """Stands in for the subprocess module inside why_report_flags only. mode: ok | spawn | exit | timeout,
    or a list of modes consumed one per call."""
    TimeoutExpired = subprocess.TimeoutExpired
    CompletedProcess = subprocess.CompletedProcess
    DEVNULL, PIPE = subprocess.DEVNULL, subprocess.PIPE

    def __init__(self):
        self.calls, self.mode = [], "ok"

    def run(self, argv, **kw):
        self.calls.append((list(argv), kw))
        m = self.mode.pop(0) if isinstance(self.mode, list) else self.mode
        if m == "spawn":
            raise FileNotFoundError(2, "No such file or directory", argv[0])
        if m == "timeout":
            raise subprocess.TimeoutExpired(argv, kw.get("timeout"))
        if m == "exit":
            return subprocess.CompletedProcess(argv, 3, b"", b"synthetic failure")
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    def messages(self):
        return [a[1] for a, _ in self.calls]


_FAKE = []


@pytest.fixture(autouse=True)
def _no_real_notify(monkeypatch):
    """Every test: why_report_flags.subprocess is a fake, so the notification script is never run."""
    fake = FakeSubprocess()
    monkeypatch.setattr(wrf, "subprocess", fake)
    _FAKE[:] = [fake]
    yield fake
    _FAKE[:] = []


def flag(model="neural_net", since="2026-11-26T01:00:00+00:00", n=3, at=None, fid=True):
    d = {"schema": "ch14_why_report_flag/1", "model_type": model, "consecutive_critical": n, "threshold": 3,
         "critical_since": since, "critical_since_known": since is not None,
         "flagged_at": at or f"2026-11-26T0{n}:00:00+00:00", "last_critical": "x"}
    if fid:
        d["flag_id"] = f"{model}@{since or d['flagged_at']}"
    return d


def write_flags(root, flags, extra_lines=()):
    p = Path(root) / wrf.FLAGS_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(f) + "\n" for f in flags) + "".join(extra_lines))
    return p


# ─────────────────────────────────────────────────────────────────────────────
# R  the reader
# ─────────────────────────────────────────────────────────────────────────────

def test_paths_match_the_writer():
    assert wrf.FLAGS_PATH == thc.WHY_REPORT_FLAGS_PATH
    assert wrf.ATTACH_PATH == thc.WHY_REPORT_ATTACH_PATH       # R1 F3: the writer reads attachments


def test_group_open_flags(tmp_path):
    old = flag("lightgbm", since=None, n=4, at="2026-11-20T00:00:00+00:00", fid=False)  # pre-flag_id line
    write_flags(tmp_path, [flag(n=3), flag(n=4), flag(n=5), old, flag("catboost", n=3)],
                extra_lines=["{not json\n"])
    rf = wrf.read_flags(str(tmp_path))
    g = rf["groups"]
    nn = "neural_net@2026-11-26T01:00:00+00:00"
    assert list(g) == [nn, "lightgbm@2026-11-20T00:00:00+00:00", "catboost@2026-11-26T01:00:00+00:00"], \
        credit("GROUP")
    assert g[nn]["repeats"] == 3 and g[nn]["max_consecutive_critical"] == 5, credit("GROUP")
    assert g["lightgbm@2026-11-20T00:00:00+00:00"]["critical_since_known"] is False, credit("GROUP")
    assert all(x["open"] for x in g.values()) and len(wrf.open_flags(str(tmp_path))) == 3, credit("GROUP")
    assert rf["bad_lines"] == {wrf.FLAGS_PATH: [6]}, credit("GROUP")       # counted, never hidden
    lines = wrf.summary_lines(str(tmp_path))
    assert any(nn in l for l in lines) and any("unreadable" in l for l in lines), credit("GROUP")
    assert wrf.read_flags(str(tmp_path / "nothing"))["groups"] == {} and wrf.summary_lines(str(tmp_path / "n")) == []


def test_attach_closes_and_is_append_only(tmp_path):
    write_flags(tmp_path, [flag(n=3), flag("catboost", n=3)])
    nn = "neural_net@2026-11-26T01:00:00+00:00"
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "why_nn.md").write_text("why")
    flags_before = (tmp_path / wrf.FLAGS_PATH).read_bytes()
    with pytest.raises(ValueError, match="unknown flag_id"):
        wrf.attach("nope@x", "docs/why_nn.md", "michael", root=str(tmp_path))
    with pytest.raises(ValueError, match="not found"):
        wrf.attach(nn, "docs/missing.md", "michael", root=str(tmp_path))
    with pytest.raises(ValueError, match="--by"):
        wrf.attach(nn, "docs/why_nn.md", " ", root=str(tmp_path))
    rec = wrf.attach(nn, "docs/why_nn.md", "michael", root=str(tmp_path), note="lr too high")
    assert rec["flag_id"] == nn and rec["by"] == "michael", credit("ATTACH")
    assert list(wrf.open_flags(str(tmp_path))) == ["catboost@2026-11-26T01:00:00+00:00"], credit("ATTACH")
    with pytest.raises(ValueError, match="already has"):
        wrf.attach(nn, "docs/why_nn.md", "michael", root=str(tmp_path))
    att1 = (tmp_path / wrf.ATTACH_PATH).read_bytes()
    # a repeat flag for an attached id stays closed; files only grow
    write_flags(tmp_path, [flag(n=3), flag("catboost", n=3), flag(n=6)])
    wrf.attach("catboost@2026-11-26T01:00:00+00:00", "docs/why_nn.md", "ai", root=str(tmp_path))
    assert (tmp_path / wrf.ATTACH_PATH).read_bytes().startswith(att1), credit("ATTACH")
    assert (tmp_path / wrf.FLAGS_PATH).read_bytes().startswith(flags_before)
    assert wrf.open_flags(str(tmp_path)) == {}, credit("ATTACH")


def test_cli_list_and_attach(tmp_path):
    write_flags(tmp_path, [flag(n=3)])
    (tmp_path / "r.md").write_text("why")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    run = lambda *a: subprocess.run([sys.executable, "-B", str(CODE_ROOT / "why_report_flags.py"),  # noqa: E731
                                     "--root", str(tmp_path), *a], capture_output=True, text=True, env=env)
    p = run("list")
    assert p.returncode == 0 and "neural_net@2026-11-26T01:00:00+00:00" in p.stdout and "OPEN" in p.stdout
    p = run("list", "--json")
    assert json.loads(p.stdout)["flags"][0]["open"] is True
    assert run("attach", "bad@id", "r.md", "--by", "m").returncode == 2
    p = run("attach", "neural_net@2026-11-26T01:00:00+00:00", "r.md", "--by", "michael")
    assert p.returncode == 0, p.stderr
    assert "no open why-report flags" in run("list").stdout, credit("ATTACH")
    assert "r.md" in run("list", "--all").stdout


def test_state_files_are_git_ignored():
    """Review fix (a): every record this package writes is git-ignored, so the first flag never dirties the
    tree and the clean-tree admission gate (scripts/gate12_cleantree_gate.py) is not tripped."""
    paths = [thc.WHY_REPORT_FLAGS_PATH, wrf.FLAGS_PATH, wrf.ATTACH_PATH, wrf.NOTIFIED_PATH]
    for p in paths:
        r = subprocess.run(["git", "-C", str(WORKTREE), "check-ignore", "-q", "--no-index", p])
        assert r.returncode == 0, credit("GITIGNORE") + f" not ignored: {p}"
        # [R1 F6] Beta Q8: --no-index passes for a force-added path; the index is checked separately
        t = subprocess.run(["git", "-C", str(WORKTREE), "ls-files", "--", p], capture_output=True, text=True)
        assert t.returncode == 0 and t.stdout.strip() == "", credit("TRACKED") + f" tracked: {p}"
    # non-vacuity of the tracked check: the tracked skip registry next to them is listed
    t = subprocess.run(["git", "-C", str(WORKTREE), "ls-files", "--", "diagnostics_outputs/model_skip_registry.json"],
                       capture_output=True, text=True)
    assert t.returncode == 0 and t.stdout.strip() == "diagnostics_outputs/model_skip_registry.json"
    # non-vacuity: the check does say "not ignored" for a neighbouring path
    r = subprocess.run(["git", "-C", str(WORKTREE), "check-ignore", "-q", "--no-index",
                        "diagnostics_outputs/why_report_flags.txt"])
    assert r.returncode == 1


def test_end_to_end_with_health_check(tmp_path, monkeypatch):
    """The writer (training_health_check, benching off by default) and the reader agree on flag_id."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "watcher_policies.json").write_text(json.dumps(
        {"training_diagnostics": {"model_skip_rules": {"max_consecutive_critical": 2}}}))
    rep = {"model_type": "neural_net", "status": "complete",
           "training_summary": {"rounds_captured": 17, "final_train_loss": 0.9, "final_val_loss": 0.95,
                                "best_val_round": 14},
           "diagnosis": {"severity": "critical", "issues": ["c"], "suggested_fixes": []}}
    os.makedirs("diagnostics_outputs", exist_ok=True)
    res = []
    for _ in range(4):
        Path("diagnostics_outputs/training_diagnostics.json").write_text(json.dumps(rep))
        res.append(thc.check_training_health())
    fids = {r["why_report_needed"]["flag_id"] for r in res[1:]}
    assert len(fids) == 1, credit("E2E")
    g = wrf.open_flags()
    assert list(g) == list(fids) and g[next(iter(fids))]["repeats"] == 3, credit("E2E")


# ─────────────────────────────────────────────────────────────────────────────
# W  the WATCHER hook
# ─────────────────────────────────────────────────────────────────────────────

def _load_watcher(name, path, block_reader=False):
    saved = sys.modules.get("why_report_flags")
    if block_reader:
        sys.modules["why_report_flags"] = None          # makes `from why_report_flags import ...` fail
    try:
        return _load(name, path)
    finally:
        if saved is not None:
            sys.modules["why_report_flags"] = saved
        else:
            sys.modules.pop("why_report_flags", None)


@pytest.fixture(scope="session")
def W():
    return _load_watcher("watcher_under_test", CODE_ROOT / "agents" / "watcher_agent.py")


@pytest.fixture(scope="session")
def W_base(tmp_path_factory):
    d = tmp_path_factory.mktemp("watcher_base")
    src = subprocess.run(["git", "-C", str(WORKTREE), "show", f"{BASE_REV}:agents/watcher_agent.py"],
                         check=True, capture_output=True).stdout
    (d / "watcher_agent.py").write_bytes(src)
    return _load_watcher("watcher_base", d / "watcher_agent.py")


class Cap(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, r):
        self.records.append((r.levelname, r.getMessage()))


HEALTH_OK = {"action": "PROCEED", "model_type": "catboost", "severity": "ok", "issues": [],
             "suggested_fixes": [], "confidence": 0.9}
HEALTH_RETRY = {"action": "RETRY", "model_type": "neural_net", "severity": "critical", "issues": ["overfit"],
                "suggested_fixes": ["lower lr"], "confidence": 0.3, "consecutive_critical": 1}


def health_flagged(fid="neural_net@2026-11-26T01:00:00+00:00", n=3):
    h = dict(HEALTH_RETRY, consecutive_critical=n)
    h["why_report_needed"] = flag(n=n) | {"flag_id": fid}
    return h


def drive(Wmod, root, healths):
    """run_pipeline(5, 6) with run_step, evaluate/execute, health check, notify and sleep faked.
    Returns what WATCHER did: run_step calls (step, params), notifications, stdout, log, return value."""
    obs = {"run_step": [], "telegram": [], "reset": [], "health_seen": []}
    fake = _FAKE[0]
    n_sent0 = len(fake.calls)
    cfg = Wmod.WatcherConfig(
        use_llm=False, manifests_dir=str(WORKTREE / "agent_manifests"),
        history_file=str(root / "watcher_history.json"), log_file=str(root / "watcher_decisions.jsonl"),
        pid_file=str(root / "watcher_daemon.pid"), daemon_state_file=str(root / "daemon_state.json"),
        pending_approval_file=str(root / "pending_approval.json"), halt_file=str(root / "halt"))
    agent = Wmod.WatcherAgent(cfg)
    agent.llm_lifecycle = agent.llm_router = None

    def run_step(step, params=None, **kw):
        obs["run_step"].append((step, json.loads(json.dumps(params, sort_keys=True, default=str))))
        return {"success": True, "confidence": 0.9}

    def evaluate(step, results, run_number):
        return types.SimpleNamespace(recommended_action="proceed", confidence=0.9), {}

    def execute(decision, context):
        agent.current_step += 1
        return True

    agent.run_step, agent.evaluate_results, agent.execute_decision = run_step, evaluate, execute
    agent._invalidate_step_freshness = lambda step: None
    hq = list(healths)
    fake_time = types.SimpleNamespace(**{k: getattr(_time, k) for k in dir(_time) if not k.startswith("_")})
    fake_time.sleep = lambda s: None
    def health(*a, **k):
        h = copy.deepcopy(hq.pop(0) if len(hq) > 1 else hq[0])
        obs["health_seen"].append(h["severity"])
        return h

    patches = dict(check_safety=lambda: True, notify_telegram=lambda m, *a, **k: obs["telegram"].append(m),
                   PROGRESS_DISPLAY_AVAILABLE=False, TRAINING_HEALTH_CHECK_AVAILABLE=True,
                   check_training_health=health,
                   reset_skip_registry=lambda m: obs["reset"].append(m), time=fake_time)
    saved = {k: Wmod.__dict__.get(k) for k in patches}
    Wmod.__dict__.update(patches)
    cap = Cap()
    lg = logging.getLogger("WatcherAgent")
    lg.addHandler(cap)
    out = io.StringIO()
    old = os.getcwd()
    os.chdir(root)
    try:
        with contextlib.redirect_stdout(out):
            obs["returned"] = agent.run_pipeline(start_step=5, end_step=6)
    finally:
        os.chdir(old)
        lg.removeHandler(cap)
        Wmod.__dict__.update(saved)
    obs["stdout"] = out.getvalue()
    obs["log"] = cap.records
    obs["telegram"] += fake.messages()[n_sent0:]   # why_report_flags' own sender (fake subprocess)
    return obs


def _no_why(obs):
    return [m for _, m in obs["log"] if TAG in m] + [m for m in obs["telegram"] if TAG in m]


@pytest.mark.parametrize("healths", [[HEALTH_OK], [HEALTH_RETRY, HEALTH_RETRY, HEALTH_RETRY, HEALTH_OK],
                                     [HEALTH_RETRY, HEALTH_RETRY, HEALTH_OK]],
                         ids=["proceed", "retry_exhausted", "retry_retry_then_healthy"])
def test_no_flags_identical_to_base(healths, W, W_base, tmp_path):
    """With no flags WATCHER does exactly what the WATCHER at BASE_REV does: the same run_step calls with
    the same params (the argv inputs of Steps 5 and 6), notifications, stdout and log."""
    (tmp_path / "new").mkdir()
    (tmp_path / "base").mkdir()
    new = drive(W, tmp_path / "new", healths)
    base = drive(W_base, tmp_path / "base", healths)
    assert [s for s, _ in new["run_step"]] == [s for s, _ in base["run_step"]], credit("WIDENT")
    assert new["run_step"] == base["run_step"], credit("WIDENT")
    assert new["telegram"] == base["telegram"] and new["stdout"] == base["stdout"], credit("WIDENT")
    assert new["log"] == base["log"] and new["returned"] == base["returned"], credit("WIDENT")
    assert _no_why(new) == []
    # non-vacuity: steps 5 and 6 ran, and the retry arm passed modified params to Step 5
    assert {5, 6} <= {s for s, _ in new["run_step"]}
    if len(healths) > 1:
        assert len(new["run_step"]) >= 3 and new["run_step"][1][1] != new["run_step"][0][1]
    if len(healths) == 3:   # [R1 F6] Beta Q8: the third health result is the healthy one and is consumed
        assert new["health_seen"] == ["critical", "critical", "ok"] == base["health_seen"], new["health_seen"]


def test_argv_builders_unchanged(W, W_base):
    """The edit touches no line of the code that builds a step's argv or the retry params."""
    for name in ("run_step", "_build_retry_params", "_get_max_training_retries"):
        assert inspect.getsource(getattr(W.WatcherAgent, name)) == \
            inspect.getsource(getattr(W_base.WatcherAgent, name)), name
    diff = subprocess.run(["git", "-C", str(WORKTREE), "diff", "--no-color", "-U0", BASE_REV, "--",
                           "agents/watcher_agent.py"], capture_output=True, text=True).stdout
    if CODE_ROOT == WORKTREE:
        added = [l for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++")]
        assert len(added) == 9, added          # 7-line import block (with comment, blank line) + 2 calls


def test_run_start_lists_open_flags_and_reminds_once(W, tmp_path):
    write_flags(tmp_path, [flag(n=3), flag(n=4), flag("catboost", n=3)])
    obs = drive(W, tmp_path, [HEALTH_OK])
    first = obs["stdout"].lstrip().splitlines()
    assert first and first[0].startswith(TAG) and "2 open why-report flag(s)" in first[0], credit("WSTART")
    assert "neural_net@2026-11-26T01:00:00+00:00" in obs["stdout"], credit("WSTART")
    logged = [m for lvl, m in obs["log"] if TAG in m and lvl == "WARNING"]
    assert any("2 open" in m for m in logged), credit("WSTART")
    rem = [m for m in obs["telegram"] if "Reminder" in m]
    assert len(rem) == 1 and "2 open" in rem[0], credit("WREMIND")
    # the run went on: both steps ran
    assert [s for s, _ in obs["run_step"]] == [5, 6]
    # attached -> no longer listed, no reminder
    (tmp_path / "w.md").write_text("why")
    for fid in list(wrf.open_flags(str(tmp_path))):
        wrf.attach(fid, "w.md", "michael", root=str(tmp_path))
    obs2 = drive(W, tmp_path, [HEALTH_OK])
    assert _no_why(obs2) == [], credit("WSTART")


def test_new_flag_notified_once_repeat_not(W, tmp_path):
    fid = "neural_net@2026-11-26T01:00:00+00:00"
    obs1 = drive(W, tmp_path, [health_flagged(fid, 3), HEALTH_OK])
    new1 = [m for m in obs1["telegram"] if "NEW flag" in m]
    assert len(new1) == 1 and fid in new1[0], credit("WNEW")
    obs2 = drive(W, tmp_path, [health_flagged(fid, 4), HEALTH_OK])
    assert [m for m in obs2["telegram"] if "NEW flag" in m] == [], credit("WNEW")
    assert any("repeat" in m for _, m in obs2["log"]), credit("WNEW")
    assert any("notification sent" in m for _, m in obs1["log"]), credit("WNEW")
    obs3 = drive(W, tmp_path, [health_flagged("catboost@2026-11-26T09:00:00+00:00", 3), HEALTH_OK])
    assert len([m for m in obs3["telegram"] if "NEW flag" in m]) == 1, credit("WNEW")
    # the health action is handled exactly as before: RETRY re-runs Step 5, then Step 6
    assert [s for s, _ in obs1["run_step"]] == [5, 5, 6]


def test_reader_failure_never_stops_the_pipeline(W, tmp_path, monkeypatch):
    write_flags(tmp_path, [flag(n=3)])

    def boom(*a, **k):
        raise RuntimeError("synthetic reader failure")
    monkeypatch.setattr(wrf, "read_flags", boom)
    monkeypatch.setattr(wrf, "flag_id_of", boom)
    try:
        obs = drive(W, tmp_path, [health_flagged(n=3), HEALTH_OK])
    except Exception as e:  # the pipeline must never stop on a reader failure
        pytest.fail(credit("WFAIL") + f" the pipeline stopped: {e!r}")
    assert [s for s, _ in obs["run_step"]] == [5, 5, 6], credit("WFAIL")
    errs = [m for lvl, m in obs["log"] if lvl == "ERROR" and TAG in m]
    assert any("run start" in m and "synthetic reader failure" in m for m in errs), credit("WFAIL")
    assert any("after the health check" in m for m in errs), credit("WFAIL")


def test_reader_import_failure_logged_pipeline_goes_on(tmp_path):
    Wb = _load_watcher("watcher_no_reader", CODE_ROOT / "agents" / "watcher_agent.py", block_reader=True)
    obs = drive(Wb, tmp_path, [health_flagged(n=3), HEALTH_OK])
    assert [s for s, _ in obs["run_step"]] == [5, 5, 6], credit("WFAIL")
    errs = [m for lvl, m in obs["log"] if lvl == "ERROR" and "flag reader not available" in m]
    assert len(errs) == 3, credit("WFAIL")             # once at run start, once per health check (2)


# ─────────────────────────────────────────────────────────────────────────────
# N  R1 F2: the sender and a failed notification (Beta Q6, Q8)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode,ok,detail", [("ok", True, "exited 0"), ("spawn", False, "could not be run"),
                                            ("exit", False, "exited 3: synthetic failure"),
                                            ("timeout", False, "timed out")])
def test_sender_outcomes(mode, ok, detail, _no_real_notify, tmp_path):
    """The real send_notification, subprocess stubbed: ok only on exit status 0; never raises."""
    fake = _no_real_notify
    fake.mode = mode
    got_ok, got_detail = wrf.send_notification("hello", root=str(tmp_path))
    assert got_ok is ok and detail in got_detail, credit("WSEND") + f" {got_ok} {got_detail}"
    argv, kw = fake.calls[-1]
    assert argv == [wrf.DEFAULT_NOTIFY_SCRIPT, "hello"], credit("WSEND")
    assert kw.get("timeout") == wrf.NOTIFY_TIMEOUT_S, credit("WSEND")


def test_sender_takes_the_policy_script(_no_real_notify, tmp_path):
    fake = _no_real_notify
    (tmp_path / "watcher_policies.json").write_text(json.dumps(
        {"notifications": {"telegram_enabled": True, "script_path": "/opt/synthetic/notify.sh"}}))
    assert wrf.send_notification("m", root=str(tmp_path))[0] is True
    assert fake.calls[-1][0][0] == "/opt/synthetic/notify.sh", credit("WSEND")
    (tmp_path / "watcher_policies.json").write_text(json.dumps({"notifications": {"telegram_enabled": False}}))
    n = len(fake.calls)
    ok, detail = wrf.send_notification("m", root=str(tmp_path))
    assert ok is False and "telegram_enabled" in detail and len(fake.calls) == n, credit("WSEND")


def _notified(root):
    return wrf._read_jsonl(os.path.join(str(root), wrf.NOTIFIED_PATH))[0]


@pytest.mark.parametrize("mode", ["spawn", "exit", "timeout"])
@pytest.mark.parametrize("retry_at", ["same_flag_health", "other_health", "run_start"])
def test_failed_send_logged_recorded_and_retried(mode, retry_at, _no_real_notify, tmp_path):
    """A failed NEW-flag notification: ERROR log, no 'notification sent', an appended 'failed' record, and
    the flag stays eligible: the next health check (same flag or any other) or run start tries again."""
    fake = _no_real_notify
    fid = "neural_net@2026-11-26T01:00:00+00:00"
    write_flags(tmp_path, [flag(n=3)])
    cap, lg = Cap(), logging.getLogger(f"why_test_{mode}_{retry_at}")
    lg.addHandler(cap)
    lg.propagate = False
    fake.mode = mode
    assert wrf.watcher_on_health(health_flagged(fid, 3), lg, root=str(tmp_path)) == "failed", credit("WRETRY")
    assert [(r["flag_id"], r["status"]) for r in _notified(tmp_path)] == [(fid, "failed")], credit("WRETRY")
    assert any(l == "ERROR" and "NOT sent" in m for l, m in cap.records), credit("WLOG")
    assert not any("notification sent" in m for _, m in cap.records), credit("WLOG")
    fake.mode = "ok"
    with contextlib.redirect_stdout(io.StringIO()):
        if retry_at == "same_flag_health":
            assert wrf.watcher_on_health(health_flagged(fid, 4), lg, root=str(tmp_path)) == "new", credit("WRETRY")
        elif retry_at == "other_health":
            assert wrf.watcher_on_health(HEALTH_OK, lg, root=str(tmp_path)) is None
        else:
            wrf.watcher_run_start(lg, root=str(tmp_path))
    assert [r["status"] for r in _notified(tmp_path)] == ["failed", "sent"], credit("WRETRY")
    assert sum("NEW flag" in m for m in fake.messages()) == 2, credit("WRETRY")
    assert any("notification sent" in m for _, m in cap.records), credit("WLOG")
    # sent now: a further check is a repeat, nothing more is sent
    n = sum("NEW flag" in m for m in fake.messages())
    assert wrf.watcher_on_health(health_flagged(fid, 5), lg, root=str(tmp_path)) == "repeat", credit("WRETRY")
    assert sum("NEW flag" in m for m in fake.messages()) == n, credit("WRETRY")


def test_watcher_send_failure_then_retry(W, _no_real_notify, tmp_path):
    """Through WATCHER's hook and the production sender (subprocess stubbed): a spawn error is not
    recorded as sent and the next run tries again."""
    fake = _no_real_notify
    fid = "neural_net@2026-11-26T01:00:00+00:00"
    write_flags(tmp_path, [flag(n=3)])
    fake.mode = "spawn"
    obs1 = drive(W, tmp_path, [health_flagged(fid, 3), HEALTH_OK])
    assert any(l == "ERROR" and "NEW flag" in m and "NOT sent" in m for l, m in obs1["log"]), credit("WLOG")
    # the flagged check fails, and the next (healthy) check of the same run tries again and fails again
    assert [r["status"] for r in _notified(tmp_path)] == ["failed", "failed"], credit("WRETRY")
    assert [s for s, _ in obs1["run_step"]] == [5, 5, 6]           # the pipeline went on
    fake.mode = "ok"
    obs2 = drive(W, tmp_path, [HEALTH_OK])
    assert "sent" in [r["status"] for r in _notified(tmp_path)], credit("WRETRY")
    assert any("NEW flag" in m for m in obs2["telegram"]), credit("WRETRY")


# ─────────────────────────────────────────────────────────────────────────────
# F  R1 F1 flag_id formula, R1 F3 a NEW flag after an attached why-report
# ─────────────────────────────────────────────────────────────────────────────

CRIT = {"model_type": "neural_net", "status": "complete",
        "training_summary": {"rounds_captured": 17, "final_train_loss": 0.9, "final_val_loss": 0.95,
                             "best_val_round": 14},
        "diagnosis": {"severity": "critical", "issues": ["c"], "suggested_fixes": []}}


def _crit():
    Path("diagnostics_outputs/training_diagnostics.json").write_text(json.dumps(CRIT))
    return thc.check_training_health()


def _policy_max3(root):
    (root / "watcher_policies.json").write_text(json.dumps(
        {"training_diagnostics": {"model_skip_rules": {"max_consecutive_critical": 3}}}))
    os.makedirs(root / "diagnostics_outputs", exist_ok=True)


@pytest.mark.parametrize("since", ["known", "unknown"])
def test_flag_id_formula_agrees_with_writer(since, tmp_path, monkeypatch):
    """R1 F1 (Beta Q5 last paragraph): for every line the writer emits, flag_id_of on the line without its
    flag_id derives the same id (critical_since known, and the count that predates the record)."""
    monkeypatch.chdir(tmp_path)
    _policy_max3(tmp_path)
    if since == "unknown":
        (tmp_path / "diagnostics_outputs/model_skip_registry.json").write_text(json.dumps(
            {"neural_net": {"consecutive_critical": 2, "last_critical": "2026-11-20T00:00:00+00:00"}}))
    for _ in range(3):
        _crit()
    lines = [json.loads(l) for l in Path(wrf.FLAGS_PATH).read_text().splitlines()]
    assert lines and all((l["critical_since"] is None) == (since == "unknown") for l in lines)
    for l in lines:
        assert wrf.flag_id_of({k: v for k, v in l.items() if k != "flag_id"}) == l["flag_id"], \
            credit("FIDFORMULA") + f" {l}"


def test_new_flag_after_attached_report_when_still_critical(_no_real_notify, tmp_path, monkeypatch):
    """R1 F3: flag, attach, two more criticals -> no flag and no notification; the third -> a NEW open flag
    linked by previous_flag_id and exactly one new notification; a fourth is a repeat of it."""
    fake = _no_real_notify
    monkeypatch.chdir(tmp_path)
    _policy_max3(tmp_path)
    lg = logging.getLogger("why_test_reflag")
    lg.addHandler(Cap())
    lg.propagate = False
    res = [_crit() for _ in range(3)]
    F = res[2]["why_report_needed"]["flag_id"]
    assert wrf.watcher_on_health(res[2], lg) == "new"
    Path("why.md").write_text("why")
    wrf.attach(F, "why.md", "michael")
    more = [_crit() for _ in range(2)]
    assert [r["action"] for r in more] == ["RETRY", "RETRY"]
    assert all("why_report_needed" not in r for r in more), credit("REFLAG_QUIET")
    assert wrf.open_flags() == {}, credit("REFLAG_QUIET")
    assert [wrf.watcher_on_health(r, lg) for r in more] == [None, None], credit("REFLAG_QUIET")
    assert sum("NEW flag" in m for m in fake.messages()) == 1, credit("REFLAG_QUIET")
    r3 = _crit()
    w = r3.get("why_report_needed") or {}
    G = w.get("flag_id")
    assert G and G != F and w.get("previous_flag_id") == F, credit("REFLAG") + f" {w}"
    assert list(wrf.open_flags()) == [G] and wrf.open_flags()[G]["previous_flag_id"] == F, credit("REFLAG")
    assert wrf.watcher_on_health(r3, lg) == "new", credit("REFLAG")
    news = [m for m in fake.messages() if "NEW flag" in m]
    assert len(news) == 2 and G in news[1] and f"Follows {F}" in news[1], credit("REFLAG")
    r4 = _crit()
    assert r4["why_report_needed"]["flag_id"] == G and wrf.watcher_on_health(r4, lg) == "repeat", credit("REFLAG")
    assert r4["consecutive_critical"] == 7       # the count is never reset by an attachment
    # control: without an attachment the same family keeps one flag_id (no new segment)
    thc.reset_skip_registry("neural_net")
    ids = {(_crit().get("why_report_needed") or {}).get("flag_id") for _ in range(5)} - {None}
    assert len(ids) == 1, ids
