#!/usr/bin/env python3
"""tests/test_owner_land.py — gates for scripts/owner_land.py.

Every gate builds its own throwaway repository with two bare remotes (`origin`, `public`) under
data/owner_land/ of the tree this file sits in. No gate touches the real repository or a real
remote. The terminal check, the typed word, the process list and the environment are injected,
so no gate needs a real terminal. Git runs for real (hooks provide the failure cases).

Run:  source ~/venvs/torch/bin/activate && PYTHONDONTWRITEBYTECODE=1 python3 -u tests/test_owner_land.py
Prints one line per gate and `COMPLETION SENTINEL: PASS` only if every gate, every mutant and
the live-tree hygiene check ran and passed.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIVE_REPO = HERE.parent
SRC = LIVE_REPO / "scripts" / "owner_land.py"
PARITY_SRC = LIVE_REPO / "scripts" / "gate12_parity_gate.py"
DATA_ROOT = LIVE_REPO / "data" / "owner_land"
TMP_ROOT: Path = None          # created by init_tmp() when the tests run, never at import

LEDGER_HEAD = """# Alpha–Beta relay ledger

Written only by `scripts/beta_relay.py`, one row per relay run that launched Codex.

| Thread | Round | Date | Request sha256 | Reply sha256 | Positions | Verdict | Other from_beta changes |
|---|---|---|---|---|---|---|---|
| X-0001 | R1 | 2026-11-12 20:19 PST | `aa` | `bb` | Q1 NO OBJECTION | REVIEW RECEIVED | — |
"""
NEW_ROW = "| X-0042 | R1 | 2026-11-13 21:00 PST | `cc` | `dd` | Q1 NO OBJECTION | REVIEW RECEIVED | — |\n"


def init_tmp() -> Path:
    global TMP_ROOT
    if TMP_ROOT is None:
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        TMP_ROOT = Path(tempfile.mkdtemp(prefix="test_", dir=DATA_ROOT))
        cfg = TMP_ROOT / "gitconfig"
        cfg.write_text("[user]\n\tname = owner-land-test\n\temail = test@invalid\n"
                       "[init]\n\tdefaultBranch = main\n[advice]\n\tdetachedHead = false\n")
        os.environ["GIT_CONFIG_GLOBAL"] = str(cfg)
        os.environ["GIT_CONFIG_NOSYSTEM"] = "1"
    return TMP_ROOT


def load_module(src_text: str, name: str):
    path = init_tmp() / f"{name}.py"
    path.write_text(src_text)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, check=True,
                          text=True).stdout.strip()


def hook(path: Path, body: str) -> None:
    path.write_text("#!/bin/sh\n" + body + "\n")
    path.chmod(0o755)


# ---------------------------------------------------------------------------------------------
# Fixture: repo (main, committed, pushed to two bare remotes), an inbox, an agent worktree.
# ---------------------------------------------------------------------------------------------
class Fx:
    n = 0

    def __init__(self):
        Fx.n += 1
        self.base = init_tmp() / f"case{Fx.n:02d}"
        self.repo = self.base / "repo"
        self.inbox = self.base / "inbox"
        self.wtbase = self.base / "worktrees"
        self.remotes = {r: self.base / f"{r}.git" for r in ("origin", "public")}
        for d in (self.repo / "scripts", self.repo / "docs/exchange/from_beta",
                  self.repo / "docs/exchange/to_beta", self.inbox, self.wtbase / "agentwt"):
            d.mkdir(parents=True)
        git(self.repo, "init", "-q", "-b", "main")
        shutil.copyfile(PARITY_SRC, self.repo / "scripts/gate12_parity_gate.py")
        (self.repo / "README.md").write_text("readme v1\n")
        (self.repo / "docs/exchange/LEDGER.md").write_text(LEDGER_HEAD)
        (self.repo / "docs/exchange/from_beta/X-0001_R1_BETA_REVIEW.md").write_text("old\n")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "base")
        for r, p in self.remotes.items():
            git(self.base, "init", "-q", "--bare", str(p))
            git(self.repo, "remote", "add", r, str(p))
            git(self.repo, "push", "-q", r, "main")
        self.head0 = git(self.repo, "rev-parse", "HEAD")
        self.calls = []
        self.lines = []
        self.asked = []
        self.answer = "land"
        self.isatty = True
        self.procs = []
        self.env = {"HOME": "/home/michael", "TERM": "xterm"}
        self.self_path = None

    # deliverables ----------------------------------------------------------------------------
    def put_inbox(self, name: str, data: bytes) -> str:
        (self.inbox / name).write_bytes(data)
        return sha(data)

    def manifest(self, rows, message="docs: land two files"):
        text = (f"message: {message}\n" if message is not None else "") + "".join(
            f"{h}  {src}  ->  {dst}  {kind}\n" for h, src, dst, kind in rows)
        (self.inbox / "MANIFEST").write_text(text)

    def std_manifest(self):
        h1 = self.put_inbox("NOTES.md", b"new notes\n")
        h2 = self.put_inbox("README.md", b"readme v2\n")
        self.manifest([(h1, "NOTES.md", "docs/new_dir/NOTES.md", "new"),
                       (h2, "README.md", "README.md", "[modify]")])
        return ["docs/new_dir/NOTES.md", "README.md"]

    def put_wt(self, rel: str, data: bytes):
        p = self.wtbase / "agentwt" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def relay_round(self):
        ex = self.repo / "docs/exchange"
        with open(ex / "LEDGER.md", "a") as f:
            f.write(NEW_ROW)
        (ex / "DEFERRED.md").write_text("deferred\n")
        (ex / "from_beta/X-0042_R1_BETA_REVIEW.md").write_text("review\n")
        (ex / "to_beta/X-0042_R1_ALPHA_REVIEW_REQUEST_t.md").write_text("request\n")
        return sorted(["docs/exchange/LEDGER.md", "docs/exchange/DEFERRED.md",
                       "docs/exchange/from_beta/X-0042_R1_BETA_REVIEW.md",
                       "docs/exchange/to_beta/X-0042_R1_ALPHA_REVIEW_REQUEST_t.md"])

    # running --------------------------------------------------------------------------------
    def deps(self, mod):
        def ask(prompt):
            self.asked.append(prompt)
            a = self.answer
            if a is EOFError:
                raise EOFError
            if callable(a):
                return a()
            return a
        return mod.Deps(isatty=lambda: self.isatty, ask=ask, processes=lambda: list(self.procs),
                        env=dict(self.env), out=self.lines.append, git_hook=self.calls.append,
                        self_pid=os.getpid(), self_path=self.self_path, inbox=self.inbox,
                        worktrees_base=self.wtbase)

    def run(self, mod, argv, cwd=None):
        self.lines.clear(); self.calls.clear(); self.asked.clear()
        return mod.run(argv, cwd or self.repo, self.deps(mod))

    @property
    def output(self) -> str:
        return "\n".join(self.lines)

    # observation ----------------------------------------------------------------------------
    def tree(self):
        out = {}
        for root, dirs, files in os.walk(self.repo):
            dirs[:] = [d for d in dirs if d != ".git"]
            r = Path(root)
            rel = r.relative_to(self.repo).as_posix()
            out["dir:" + rel] = None
            for f in files:
                p = r / f
                out[p.relative_to(self.repo).as_posix()] = (sha(p.read_bytes()),
                                                            stat.S_IMODE(p.stat().st_mode))
        return out

    def remote_head(self, r):
        return git(self.base, f"--git-dir={self.remotes[r]}", "rev-parse", "refs/heads/main")

    def state(self):
        return {"head": git(self.repo, "rev-parse", "HEAD"),
                "status": git(self.repo, "status", "--porcelain", "--untracked-files=all"),
                "index": git(self.repo, "diff", "--cached", "--name-only"),
                "tree": self.tree(),
                "origin": self.remote_head("origin"), "public": self.remote_head("public")}

    def landed_ok(self, rc, expect_paths):
        head = git(self.repo, "rev-parse", "HEAD")
        files = sorted(git(self.repo, "show", "--name-only", "--format=", "HEAD").splitlines())
        parent = git(self.repo, "rev-parse", "HEAD~1")
        checks = {
            "rc=0": rc == 0,
            "one new commit": parent == self.head0 and head != self.head0,
            "origin at commit": self.remote_head("origin") == head,
            "public at commit": self.remote_head("public") == head,
            "tree clean": git(self.repo, "status", "--porcelain", "--untracked-files=all") == "",
            "commit = plan": files == sorted(expect_paths),
        }
        bad = [k for k, v in checks.items() if not v]
        return not bad, (f"rc={rc} files={files}" if not bad else f"FAILED {bad} rc={rc} "
                         f"files={files} out={self.output[-600:]!r}")


def refused_unchanged(fx, mod, argv, rc_want, needles, cwd=None, before=None):
    before = before or fx.state()
    rc = fx.run(mod, argv, cwd)
    after = fx.state()
    missing = [n for n in needles if n not in fx.output]
    ok = rc == rc_want and before == after and not missing
    return ok, (f"rc={rc} unchanged={before == after} named={not missing}" if ok else
                f"rc={rc} want {rc_want}; unchanged={before == after}; missing {missing}; "
                f"out={fx.output[-700:]!r}")


# ---------------------------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------------------------
def g_clean_manifest(mod):
    fx = Fx(); paths = fx.std_manifest()
    rc = fx.run(mod, ["--manifest"])
    ok, d = fx.landed_ok(rc, paths)
    ok = ok and (fx.repo / "README.md").read_bytes() == b"readme v2\n"
    return ok, d


def g_clean_worktree(mod):
    fx = Fx()
    fx.put_wt("README.md", b"readme from agent\n")
    fx.put_wt("tests/test_new.py", b"print('x')\n")
    rc = fx.run(mod, ["--worktree", "agentwt", "README.md", "tests/test_new.py", "-m", "tests: x"])
    ok, d = fx.landed_ok(rc, ["README.md", "tests/test_new.py"])
    msg = git(fx.repo, "log", "-1", "--format=%s")
    return ok and msg == "tests: x", d + f" msg={msg!r}"


def g_clean_exchange(mod):
    fx = Fx(); paths = fx.relay_round()
    rc = fx.run(mod, ["--exchange", "X-0042"])
    ok, d = fx.landed_ok(rc, paths)
    msg = git(fx.repo, "log", "-1", "--format=%s")
    want = "exchange: X-0042 R1 (REVIEW RECEIVED)"
    return ok and msg == want, d + f" msg={msg!r}"


def g_not_tty(mod):
    fx = Fx(); fx.std_manifest(); fx.isatty = False
    ok, d = refused_unchanged(fx, mod, ["--manifest"], mod.EXIT_REFUSED, ["NOT-A-TERMINAL"])
    return ok and not fx.asked, d + f" asked={len(fx.asked)}"


def g_confirm(mod):
    res = []
    for answer in ("yes", "Land", "", "land now", EOFError):
        fx = Fx(); fx.std_manifest(); fx.answer = answer
        ok, d = refused_unchanged(fx, mod, ["--manifest"], mod.EXIT_ABORTED, ["aborted"])
        res.append((ok and len(fx.asked) == 1, f"{answer!r}: {d}"))
    return all(r for r, _ in res), "; ".join(d for _, d in res)


def g_dirty(mod):
    fx = Fx(); fx.std_manifest()
    (fx.repo / "stray_notes.txt").write_text("not in the plan\n")
    return refused_unchanged(fx, mod, ["--manifest"], mod.EXIT_REFUSED,
                             ["DIRTY-PATH", "stray_notes.txt"])


def g_dirty_tracked(mod):
    fx = Fx(); fx.std_manifest()
    (fx.repo / "scripts/gate12_parity_gate.py").write_text("# edited\n")
    return refused_unchanged(fx, mod, ["--manifest"], mod.EXIT_REFUSED,
                             ["DIRTY-PATH", "scripts/gate12_parity_gate.py"])


def g_sha(mod):
    fx = Fx(); fx.std_manifest()
    (fx.inbox / "NOTES.md").write_bytes(b"tampered\n")
    ok, d = refused_unchanged(fx, mod, ["--manifest"], mod.EXIT_REFUSED, ["SHA-MISMATCH",
                                                                         "NOTES.md"])
    return ok and not (fx.repo / "docs/new_dir").exists(), d


def g_new_exists(mod):
    fx = Fx()
    h = fx.put_inbox("README.md", b"readme v2\n")
    fx.manifest([(h, "README.md", "README.md", "new")])
    return refused_unchanged(fx, mod, ["--manifest"], mod.EXIT_REFUSED, ["NEW-EXISTS", "README.md"])


def g_modify_missing(mod):
    fx = Fx()
    h = fx.put_inbox("X.md", b"x\n")
    fx.manifest([(h, "X.md", "docs/absent.md", "modify")])
    return refused_unchanged(fx, mod, ["--manifest"], mod.EXIT_REFUSED,
                             ["MODIFY-MISSING", "docs/absent.md"])


def g_source_missing(mod):
    fx = Fx()
    fx.manifest([("0" * 64, "GONE.md", "docs/gone.md", "new")])
    ok1, d1 = refused_unchanged(fx, mod, ["--manifest"], mod.EXIT_REFUSED, ["SOURCE-MISSING"])
    fx2 = Fx()
    ok2, d2 = refused_unchanged(fx2, mod, ["--worktree", "agentwt", "docs/none.md", "-m", "x"],
                                mod.EXIT_REFUSED, ["SOURCE-MISSING"])
    return ok1 and ok2, f"manifest: {d1}; worktree: {d2}"


def g_governed(mod):
    fx = Fx()
    dests = ["miner/new_thing.py", "sieve_gpu_worker.py", "prng_registry.py",
             "agents/watcher_agent.py", "gate12_launch.sh", "agent_manifests/x.json",
             "scripts/gate12_parity_gate.py"]
    rows = []
    for i, dst in enumerate(dests):
        h = fx.put_inbox(f"f{i}", f"content {i}\n".encode())
        rows.append((h, f"f{i}", dst, "modify" if dst == "scripts/gate12_parity_gate.py" else "new"))
    fx.manifest(rows)
    return refused_unchanged(fx, mod, ["--manifest"], mod.EXIT_REFUSED,
                             [f"GOVERNED: {d} is" for d in dests])


def g_linked_worktree(mod):
    fx = Fx()
    git(fx.repo, "checkout", "-q", "-b", "side")
    linked = fx.base / "linked"
    git(fx.repo, "worktree", "add", "-q", str(linked), "main")
    fx.put_wt("README.md", b"r3\n")
    return refused_unchanged(fx, mod, ["--worktree", "agentwt", "README.md", "-m", "x"],
                             mod.EXIT_REFUSED, ["LINKED-WORKTREE"], cwd=linked)


def g_branch(mod):
    fx = Fx()
    git(fx.repo, "checkout", "-q", "-b", "feature")
    fx.put_wt("README.md", b"r3\n")
    return refused_unchanged(fx, mod, ["--worktree", "agentwt", "README.md", "-m", "x"],
                             mod.EXIT_REFUSED, ["BRANCH: the current branch is feature"])


def g_proc(mod):
    res = []
    for argv in (["python3", "-u", "scripts/beta_relay.py", "--request", "r.md"],
                 ["/home/michael/venvs/torch/bin/python3", "miner/range_miner_coordinator.py"],
                 ["python3", "-m", "miner.range_miner_coordinator"],
                 ["python3", "agents/watcher_agent.py", "--run-pipeline"],
                 ["python3", "-u", "window_optimizer.py", "--trials", "5"],
                 ["./coordinator.py"]):
        fx = Fx(); fx.std_manifest(); fx.procs = [(99999, argv)]
        ok, d = refused_unchanged(fx, mod, ["--manifest"], mod.EXIT_REFUSED,
                                  ["PROCESS-RUNNING", "pid 99999"])
        res.append((ok, f"{argv[-1] if len(argv) < 3 else argv[2]}: {'refused' if ok else d}"))
    return all(r for r, _ in res), "; ".join(d for _, d in res)


def g_proc_decoys(mod):
    """A shell whose -c text names the relay (the pgrep self-match), an editor open on
    watcher_agent.py, and this test process itself must NOT refuse."""
    fx = Fx(); paths = fx.std_manifest()
    fx.procs = [(11, ["/bin/bash", "-c", "pgrep -af beta_relay.py; echo done"]),
                (12, ["vim", "agents/watcher_agent.py"]),
                (13, ["less", "coordinator.py"]),
                (os.getpid(), ["python3", "scripts/beta_relay.py"])]   # own pid is skipped
    rc = fx.run(mod, ["--manifest"])
    return fx.landed_ok(rc, paths)


def g_push2(mod):
    fx = Fx(); fx.std_manifest()
    hook(fx.remotes["public"] / "hooks/pre-receive", "echo 'public rejects' >&2; exit 1")
    rc = fx.run(mod, ["--manifest"])
    head = git(fx.repo, "rev-parse", "HEAD")
    want_cmd = f"cd {fx.repo} && git push public main"
    checks = {"rc=EXIT_PARTIAL": rc == mod.EXIT_PARTIAL and rc != 0,
              "local commit kept": head != fx.head0 and git(fx.repo, "rev-parse", "HEAD~1") == fx.head0,
              "origin advanced": fx.remote_head("origin") == head,
              "public not advanced": fx.remote_head("public") == fx.head0,
              "finishing command printed": want_cmd in fx.output}
    bad = [k for k, v in checks.items() if not v]
    return not bad, f"rc={rc} " + ("all hold" if not bad else f"FAILED {bad} out={fx.output[-500:]!r}")


def g_rollback(mod):
    fx = Fx(); fx.std_manifest()
    hook(fx.repo / ".git/hooks/pre-commit", "exit 1")
    before = fx.state()
    rc = fx.run(mod, ["--manifest"])
    after = fx.state()
    ok = (rc == mod.EXIT_ROLLED_BACK and before == after and "rolled back" in fx.output
          and not (fx.repo / "docs/new_dir").exists() and len(fx.asked) == 1)
    diff = {k for k in before if before[k] != after[k]}
    return ok, f"rc={rc} unchanged={before == after} differs={sorted(diff)}" + (
        "" if ok else f" out={fx.output[-500:]!r}")


def g_rollback_exchange(mod):
    """A failed exchange landing must unstage and must never delete Beta's files."""
    fx = Fx(); fx.relay_round()
    hook(fx.repo / ".git/hooks/pre-commit", "exit 1")
    before = fx.state()
    rc = fx.run(mod, ["--exchange", "X-0042"])
    after = fx.state()
    ok = rc == mod.EXIT_ROLLED_BACK and before == after
    return ok, f"rc={rc} unchanged={before == after}"


def g_add_only(mod):
    res = []
    for setup, argv, label in (
            (lambda fx: fx.std_manifest(), ["--manifest"], "manifest"),
            (lambda fx: fx.relay_round(), ["--exchange", "X-0042"], "exchange")):
        fx = Fx(); paths = setup(fx)
        rc = fx.run(mod, argv)
        adds = [c for c in fx.calls if c and c[0] == "add"]
        ok = (rc == 0 and len(adds) == len(paths)
              and all(len(a) == 3 and a[1] == "--" for a in adds)
              and sorted(a[2] for a in adds) == sorted(paths))
        res.append((ok, f"{label}: add calls={adds}"))
    return all(r for r, _ in res), "; ".join(d for _, d in res)


def g_message(mod):
    fx = Fx(); fx.put_wt("README.md", b"r3\n")
    ok1, d1 = refused_unchanged(fx, mod, ["--worktree", "agentwt", "README.md", "-m", "  "],
                                mod.EXIT_REFUSED, ["MESSAGE: the commit message is empty"])
    fx2 = Fx(); h = fx2.put_inbox("A", b"a\n"); fx2.manifest([(h, "A", "A.md", "new")], message=None)
    ok2, d2 = refused_unchanged(fx2, mod, ["--manifest"], mod.EXIT_REFUSED,
                                ["no `message:` line", "MESSAGE: the commit message is empty"])
    return ok1 and ok2, f"worktree: {d1}; manifest: {d2}"


def g_remote(mod):
    fx = Fx(); fx.std_manifest()
    git(fx.repo, "remote", "remove", "public")
    return refused_unchanged(fx, mod, ["--manifest"], mod.EXIT_REFUSED,
                             ["REMOTE: remote `public` is not configured"])


def g_agent_env(mod):
    res = []
    for k in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "AI_AGENT", "CODEX_SANDBOX"):
        fx = Fx(); fx.std_manifest(); fx.env[k] = "1"
        ok, d = refused_unchanged(fx, mod, ["--manifest"], mod.EXIT_REFUSED,
                                  ["AGENT-ENVIRONMENT", k])
        res.append((ok, f"{k}: {'refused' if ok else d}"))
    return all(r for r, _ in res), "; ".join(d for _, d in res)


def g_exchange_guards(mod):
    fx = Fx()                                       # no relay round at all
    ok1, d1 = refused_unchanged(fx, mod, ["--exchange", "X-0001"], mod.EXIT_REFUSED,
                                ["LEDGER.md is not modified", "already at HEAD"])
    fx2 = Fx(); fx2.relay_round()                   # another thread's file is left over
    (fx2.repo / "docs/exchange/from_beta/X-00420_R1_BETA_REVIEW.md").write_text("other\n")
    ok2, d2 = refused_unchanged(fx2, mod, ["--exchange", "X-0042"], mod.EXIT_REFUSED,
                                ["DIRTY-PATH", "X-00420_R1_BETA_REVIEW.md"])
    return ok1 and ok2, f"no round: {d1}; other thread: {d2}"


def g_recheck(mod):
    """The tree changes while the owner reads the plan: the second check refuses."""
    fx = Fx(); fx.std_manifest()
    before = fx.state()

    def sneak():
        (fx.repo / "late.txt").write_text("arrived after the plan\n")
        return "land"
    fx.answer = sneak
    rc = fx.run(mod, ["--manifest"])
    after = fx.state()
    after["tree"].pop("late.txt", None)
    ok = (rc == mod.EXIT_REFUSED and "DIRTY-PATH" in fx.output and "late.txt" in fx.output
          and after["head"] == before["head"] and after["tree"] == before["tree"]
          and after["origin"] == before["origin"] and after["public"] == before["public"])
    return ok, f"rc={rc} head unchanged={after['head'] == before['head']}"


def g_self(mod):
    fx = Fx(); fx.std_manifest()
    fx.self_path = fx.base / "elsewhere" / "owner_land.py"
    return refused_unchanged(fx, mod, ["--manifest"], mod.EXIT_REFUSED, ["SELF: this copy is"])


def g_path_safety(mod):
    fx = Fx()
    h = fx.put_inbox("A", b"a\n")
    fx.manifest([(h, "A", "../escape.md", "new"), (h, "A", ".git/config", "modify"),
                 (h, "../A", "docs/a.md", "new")])
    return refused_unchanged(fx, mod, ["--manifest"], mod.EXIT_REFUSED,
                             ["'../escape.md' is not a safe", "'.git/config' is not a safe",
                              "outside the inbox"])


GATES = [
    ("G-CLEAN-MANIFEST", g_clean_manifest), ("G-CLEAN-WORKTREE", g_clean_worktree),
    ("G-CLEAN-EXCHANGE", g_clean_exchange), ("G-NOTTY", g_not_tty), ("G-CONFIRM", g_confirm),
    ("G-DIRTY", g_dirty), ("G-DIRTY-TRACKED", g_dirty_tracked), ("G-SHA", g_sha),
    ("G-NEW-EXISTS", g_new_exists), ("G-MODIFY-MISSING", g_modify_missing),
    ("G-SOURCE-MISSING", g_source_missing), ("G-GOVERNED", g_governed),
    ("G-LINKED-WT", g_linked_worktree), ("G-BRANCH", g_branch), ("G-PROC", g_proc),
    ("G-PROC-DECOYS", g_proc_decoys), ("G-PUSH2", g_push2), ("G-ROLLBACK", g_rollback),
    ("G-ROLLBACK-EXCH", g_rollback_exchange), ("G-ADD-ONLY", g_add_only),
    ("G-MESSAGE", g_message), ("G-REMOTE", g_remote), ("G-AGENT-ENV", g_agent_env),
    ("G-EXCHANGE-GUARDS", g_exchange_guards), ("G-RECHECK", g_recheck), ("G-SELF", g_self),
    ("G-PATH-SAFETY", g_path_safety),
]

# (name, description, anchor (must occur exactly once), replacement, gates that must go red)
MUTANTS = [
    ("M-TTY", "terminal check removed",
     "    probs += check_terminal(deps)\n", "", ["G-NOTTY"]),
    ("M-CONFIRM", "confirmation removed",
     "    if answer.strip() != CONFIRM_WORD:\n", "    if False:\n", ["G-CONFIRM"]),
    ("M-DIRTY", "dirty-path check removed",
     "    probs += check_dirty(plan, porcelain(git))\n", "", ["G-DIRTY", "G-DIRTY-TRACKED"]),
    ("M-ADD-A", "explicit paths replaced by -A",
     '            git.run("add", "--", path)\n', '            git.run("add", "-A")\n',
     ["G-ADD-ONLY"]),
    ("M-PUSH2", "second push removed",
     "    for i, r in enumerate(REMOTES):\n", "    for i, r in enumerate(REMOTES[:1]):\n",
     ["G-CLEAN-MANIFEST", "G-PUSH2"]),
]


def run_gate(mod, name):
    fn = dict(GATES)[name]
    try:
        return fn(mod)
    except Exception as e:                       # noqa: BLE001 — a crash is a red gate
        return False, f"EXCEPTION {type(e).__name__}: {e}"


def live_state():
    main_git = Path(git(LIVE_REPO, "rev-parse", "--git-common-dir"))
    if not main_git.is_absolute():
        main_git = LIVE_REPO / main_git
    main = main_git.resolve().parent
    snap = {}
    for label, d in (("main", main), ("this", LIVE_REPO)):
        snap[label] = (git(d, "rev-parse", "HEAD"),
                       git(d, "status", "--porcelain", "--untracked-files=all", "--",
                           ".", ":(exclude)data/owner_land"),
                       git(d, "for-each-ref", "--format=%(refname) %(objectname)"))
    return main, snap


def main() -> int:
    init_tmp()
    src = SRC.read_text()
    mod = load_module(src, "owner_land_under_test")
    main_repo, live_before = live_state()
    print(f"  temp={TMP_ROOT}  live main checkout={main_repo}", flush=True)

    results = []
    for name, _ in GATES:
        ok, detail = run_gate(mod, name)
        results.append((name, ok))
        print(f"  [{'PASS' if ok else 'FAIL':<5}] {name:<18} — {detail}", flush=True)

    for mname, desc, anchor, repl, targets in MUTANTS:
        count = src.count(anchor)
        if count != 1:
            results.append((mname, False))
            print(f"  [ERROR] {mname:<10} — mutant did not apply ({desc}): anchor found {count}x",
                  flush=True)
            continue
        mm = load_module(src.replace(anchor, repl), f"owner_land_{mname.replace('-', '_')}")
        reds = {t: not run_gate(mm, t)[0] for t in targets}
        ok = all(reds.values())
        results.append((mname, ok))
        print(f"  [{'PASS' if ok else 'FAIL':<5}] {mname:<10} — {desc}: "
              + ", ".join(f"{t} {'red' if v else 'GREEN (not detected)'}" for t, v in reds.items()),
              flush=True)

    _, live_after = live_state()
    ok = live_before == live_after
    results.append(("HYG-LIVE", ok))
    print(f"  [{'PASS' if ok else 'FAIL':<5}] HYG-LIVE           — main checkout and this tree: "
          f"HEAD, status and refs unchanged by the suite: {ok}", flush=True)

    passed = sum(ok for _, ok in results)
    print(f"  {passed}/{len(results)} PASS", flush=True)
    if passed == len(results) == len(GATES) + len(MUTANTS) + 1:
        shutil.rmtree(TMP_ROOT, ignore_errors=True)
        if DATA_ROOT.exists() and not any(DATA_ROOT.iterdir()):
            DATA_ROOT.rmdir()
        print("COMPLETION SENTINEL: PASS", flush=True)
        return 0
    print(f"  temp kept for inspection: {TMP_ROOT}", flush=True)
    print("COMPLETION SENTINEL: FAIL", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
