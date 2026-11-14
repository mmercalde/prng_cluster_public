#!/usr/bin/env python3
"""scripts/owner_land.py — the owner's landing script.

ONLY MICHAEL RUNS THIS. No agent (Claude Code, Codex, any other) runs it, in any mode, against
any repository. It is a path to `git commit` and `git push` that an agent deny-list does not
see, so it guards itself: it refuses when standard input is not a terminal, refuses when it
sees an agent's environment, and lands nothing until a person types the word `land`. There is
no flag that skips the confirmation.

WHAT IT DOES
------------
It lands a named list of files as ONE commit on `main` of the main checkout and pushes that
commit to BOTH remotes, `origin` then `public` (CLAUDE.md §1 rules 1-2). Three ways to give it
the list:

  manifest  files Alpha delivered, copied with a manifest into an inbox outside the repo:
              message: <one-line commit message>
              <sha256>  <file in the inbox>  ->  <repo-relative destination>  new|modify
            (`[new]` / `[modify]` are accepted too). Blank lines and `#` lines are ignored.
            Each file's sha256 is verified before anything is copied.

  worktree  an agent's output: copies the named repo-relative paths from
            ~/tfm_worktrees/<name>/ to the same paths in the main tree.
            `new` / `modify` is decided by whether the path is tracked at HEAD.

  exchange  a relay round: lands docs/exchange/LEDGER.md, docs/exchange/DEFERRED.md if it
            changed, and the untracked files in docs/exchange/from_beta/ and
            docs/exchange/to_beta/ whose names start with `<thread>_`. The message is built
            from the ledger's last row for the thread: `exchange: <thread> <round> (<verdict>)`.

ORDER, IN EVERY MODE
--------------------
1. Print the plan (each file, new/modify, sha256, the commit message).
2. Refuse, changing nothing, on any failed check (each has its own message; all are printed).
3. Ask for the word `land`. Anything else aborts and changes nothing.
4. Re-run every check (the tree may have changed while the owner read the plan).
5. Copy the files, `git add -- <path>` once per planned path (never -A, never .), commit.
6. `git push origin main`, then `git push public main`.
7. Print `git status --porcelain` and `git log --oneline -1`.
A failure after the confirmation and before the commit exists restores every copied file to its
exact prior bytes (new files removed), unstages the planned paths, and says so.

EXIT CODES
----------
0 landed and pushed to both remotes · 2 usage error (argparse) · 3 refused, nothing changed ·
4 aborted at the confirmation, nothing changed · 5 failed after the confirmation, rolled back ·
6 committed locally, a push failed (the finishing command is printed).

USAGE (from ser8; `-t` gives the remote side a terminal, which the script requires)
-----
  ssh -t michael@192.168.3.177 'cd /home/michael/distributed_prng_analysis && \\
      python3 scripts/owner_land.py --manifest'
  ssh -t ... 'cd ... && python3 scripts/owner_land.py --worktree <name> <path> [<path> ...] -m "<msg>"'
  ssh -t ... 'cd ... && python3 scripts/owner_land.py --exchange <thread-id>'

Standard library only.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------------------------
# Constants. Every one traces to the brief (docs/CLAUDE_CODE_INSTRUCTIONS_OWNER_LAND_SCRIPT_
# 20261113.md §1) or to CLAUDE.md §1 rules 1-2; none is a measured value.
# ---------------------------------------------------------------------------------------------
CONFIRM_WORD = "land"
BRANCH = "main"
REMOTES = ("origin", "public")                       # pushed in this order
INBOX_DEFAULT = Path.home() / "tfm_inbox"
MANIFEST_NAME = "MANIFEST"
WORKTREES_BASE = Path.home() / "tfm_worktrees"
SELF_REL = "scripts/owner_land.py"

PARITY_GATE_REL = "scripts/gate12_parity_gate.py"     # where GOVERNED_FILES is defined
# Landed by hand only. GOVERNED_FILES is read from PARITY_GATE_REL at run time, never copied.
HAND_LANDED_FILES = ("agents/watcher_agent.py", "gate12_launch.sh", PARITY_GATE_REL)
HAND_LANDED_DIRS = ("miner/", "agent_manifests/")

EXCHANGE_DIR = "docs/exchange"
LEDGER_REL = f"{EXCHANGE_DIR}/LEDGER.md"
DEFERRED_REL = f"{EXCHANGE_DIR}/DEFERRED.md"
EXCHANGE_DROP_DIRS = (f"{EXCHANGE_DIR}/from_beta", f"{EXCHANGE_DIR}/to_beta")
LEDGER_COL_THREAD, LEDGER_COL_ROUND, LEDGER_COL_VERDICT = 0, 1, 6   # LEDGER.md header order

# Environment variables that mark an agent session. Claude Code sets CLAUDECODE=1, AI_AGENT and
# CLAUDE_CODE_*; Codex sets CODEX_*.
AGENT_ENV_EXACT = ("CLAUDECODE", "AI_AGENT")
AGENT_ENV_PREFIXES = ("CLAUDE_CODE_", "CODEX_")

EXIT_LANDED, EXIT_REFUSED, EXIT_ABORTED, EXIT_ROLLED_BACK, EXIT_PARTIAL = 0, 3, 4, 5, 6

THREAD_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]*$")
MANIFEST_ROW_RE = re.compile(
    r"^(?P<sha>[0-9a-fA-F]{64})\s+(?P<src>\S+)\s+->\s+(?P<dst>\S+)\s+\[?(?P<kind>new|modify)\]?$")


# ---------------------------------------------------------------------------------------------
# Injectable dependencies. Tests replace these; the defaults are the real machine.
# ---------------------------------------------------------------------------------------------
def default_processes() -> List[Tuple[int, List[str]]]:
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            raw = (d / "cmdline").read_bytes()
        except OSError:
            continue
        argv = [a.decode("utf-8", "replace") for a in raw.split(b"\0") if a]
        if argv:
            out.append((int(d.name), argv))
    return out


@dataclass
class Deps:
    isatty: Callable[[], bool] = lambda: sys.stdin.isatty()
    ask: Callable[[str], str] = input
    processes: Callable[[], List[Tuple[int, List[str]]]] = default_processes
    env: Dict[str, str] = field(default_factory=lambda: dict(os.environ))
    out: Callable[[str], None] = lambda s: print(s, flush=True)
    git_hook: Optional[Callable[[List[str]], None]] = None   # observer: called with each git argv
    self_pid: int = field(default_factory=os.getpid)
    self_path: Optional[Path] = None          # the running script; None = do not check identity
    inbox: Path = INBOX_DEFAULT
    worktrees_base: Path = WORKTREES_BASE


class Git:
    def __init__(self, repo: Path, deps: Deps):
        self.repo, self.deps = repo, deps

    def run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        argv = list(args)
        if self.deps.git_hook is not None:
            self.deps.git_hook(argv)
        cp = subprocess.run(["git", *argv], cwd=str(self.repo), capture_output=True,
                            stdin=subprocess.DEVNULL)
        if check and cp.returncode != 0:
            raise GitError(argv, cp)
        return cp

    def text(self, *args: str) -> str:
        return self.run(*args).stdout.decode("utf-8", "replace").strip()

    def head_blob(self, rel: str) -> Optional[bytes]:
        cp = self.run("cat-file", "blob", f"HEAD:{rel}", check=False)
        return cp.stdout if cp.returncode == 0 else None


class GitError(RuntimeError):
    def __init__(self, argv, cp):
        err = cp.stderr.decode("utf-8", "replace").strip()
        super().__init__(f"git {' '.join(argv)} exited {cp.returncode}: {err}")
        self.cp = cp


# ---------------------------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------------------------
@dataclass
class Entry:
    dest: str                         # repo-relative POSIX
    kind: str                         # "new" | "modify"
    source: Optional[Path]            # None = already in place (exchange mode)
    sha: Optional[str] = None         # sha256 of the bytes that will land; None = unreadable
    expected_sha: Optional[str] = None  # manifest mode only


@dataclass
class Plan:
    mode: str
    message: str
    entries: List[Entry] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)   # found while building the plan

    @property
    def paths(self) -> List[str]:
        return [e.dest for e in self.entries]


def sha256_file(p: Path) -> Optional[str]:
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError:
        return None


def safe_rel(rel: str) -> Optional[str]:
    """Normalised repo-relative POSIX path, or None if it escapes or touches .git."""
    if not rel or rel.startswith("/") or "\\" in rel:
        return None
    parts = PurePosixPath(rel).parts
    if any(p in ("..", "") for p in parts) or parts[0] == ".git" or rel.endswith("/"):
        return None
    return str(PurePosixPath(*parts))


def kind_from_head(git: Git, rel: str) -> str:
    return "modify" if git.head_blob(rel) is not None else "new"


def build_manifest_plan(git: Git, manifest: Path) -> Plan:
    plan = Plan(mode="manifest", message="")
    try:
        lines = manifest.read_text().splitlines()
    except OSError as e:
        plan.problems.append(f"MANIFEST: cannot read {manifest}: {e}")
        return plan
    base = manifest.parent.resolve()
    seen_msg = False
    for n, line in enumerate(lines, 1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("message:"):
            if seen_msg:
                plan.problems.append(f"MANIFEST: line {n}: a second `message:` line")
            plan.message, seen_msg = s[len("message:"):].strip(), True
            continue
        m = MANIFEST_ROW_RE.match(s)
        if not m:
            plan.problems.append(f"MANIFEST: line {n} is not `<sha256>  <file>  ->  <dest>  "
                                 f"new|modify`: {s!r}")
            continue
        dest = safe_rel(m["dst"])
        if dest is None:
            plan.problems.append(f"MANIFEST: line {n}: destination {m['dst']!r} is not a safe "
                                 f"repo-relative path")
            continue
        src = (base / m["src"]).resolve()
        if base not in src.parents:
            plan.problems.append(f"MANIFEST: line {n}: source {m['src']!r} is outside the inbox")
            continue
        plan.entries.append(Entry(dest=dest, kind=m["kind"], source=src, sha=sha256_file(src),
                                  expected_sha=m["sha"].lower()))
    if not seen_msg:
        plan.problems.append("MANIFEST: no `message:` line")
    return plan


def build_worktree_plan(git: Git, base: Path, name: str, paths: Sequence[str],
                        message: str) -> Plan:
    plan = Plan(mode="worktree", message=message)
    if not THREAD_RE.match(name):
        plan.problems.append(f"WORKTREE: name {name!r} is not a plain directory name")
        return plan
    root = base / name
    if not root.is_dir():
        plan.problems.append(f"WORKTREE: {root} does not exist")
    for raw in paths:
        dest = safe_rel(raw)
        if dest is None:
            plan.problems.append(f"WORKTREE: path {raw!r} is not a safe repo-relative path")
            continue
        src = root / dest
        plan.entries.append(Entry(dest=dest, kind=kind_from_head(git, dest), source=src,
                                  sha=sha256_file(src)))
    return plan


def parse_ledger_rows(text: str) -> List[List[str]]:
    rows = []
    for line in text.splitlines():
        s = line.strip()
        if not (s.startswith("|") and s.endswith("|")):
            continue
        cells = [c.strip() for c in s[1:-1].split("|")]
        if not cells or cells[0] in ("Thread", "") or set(cells[0]) <= set("-: "):
            continue
        rows.append(cells)
    return rows


def build_exchange_plan(git: Git, thread: str, status: Dict[str, str]) -> Plan:
    plan = Plan(mode="exchange", message="")
    if not THREAD_RE.match(thread):
        plan.problems.append(f"EXCHANGE: thread id {thread!r} is not a plain id")
        return plan
    repo = git.repo
    if LEDGER_REL not in status:
        plan.problems.append(f"EXCHANGE: {LEDGER_REL} is not modified; the relay left nothing "
                             f"to land")
    rels = [LEDGER_REL] if LEDGER_REL in status else []
    if DEFERRED_REL in status:
        rels.append(DEFERRED_REL)
    for rel, code in sorted(status.items()):
        d, _, base = rel.rpartition("/")
        if d in EXCHANGE_DROP_DIRS and base.startswith(thread + "_") and code == "??":
            rels.append(rel)
    for rel in rels:
        plan.entries.append(Entry(dest=rel, kind=kind_from_head(git, rel), source=None,
                                  sha=sha256_file(repo / rel)))
    # Message from the ledger's last row for the thread; that row must be new since HEAD.
    try:
        rows = [r for r in parse_ledger_rows((repo / LEDGER_REL).read_text())
                if r[LEDGER_COL_THREAD] == thread]
    except OSError as e:
        plan.problems.append(f"EXCHANGE: cannot read {LEDGER_REL}: {e}")
        return plan
    if not rows:
        plan.problems.append(f"EXCHANGE: {LEDGER_REL} has no row for thread {thread}")
        return plan
    last = rows[-1]
    if len(last) <= LEDGER_COL_VERDICT:
        plan.problems.append(f"EXCHANGE: the last ledger row for {thread} has {len(last)} cells")
        return plan
    head = git.head_blob(LEDGER_REL)
    head_rows = parse_ledger_rows(head.decode("utf-8", "replace")) if head else []
    if last in head_rows:
        plan.problems.append(f"EXCHANGE: the last ledger row for {thread} "
                             f"({last[LEDGER_COL_ROUND]}) is already at HEAD; no new round")
    plan.message = (f"exchange: {thread} {last[LEDGER_COL_ROUND]} "
                    f"({last[LEDGER_COL_VERDICT]})")
    return plan


# ---------------------------------------------------------------------------------------------
# Checks. Each returns a list of refusal messages; empty = pass.
# ---------------------------------------------------------------------------------------------
def porcelain(git: Git) -> Dict[str, str]:
    """path -> two-char status, every untracked file listed individually; renames give both."""
    raw = git.run("status", "--porcelain", "-z", "--untracked-files=all").stdout
    toks = raw.split(b"\0")
    out: Dict[str, str] = {}
    i = 0
    while i < len(toks):
        t = toks[i].decode("utf-8", "surrogateescape")
        i += 1
        if len(t) < 4:
            continue
        code, path = t[:2], t[3:]
        out[path] = code
        if code[0] in "RC" and i < len(toks):          # -z: the origin path follows
            out[toks[i].decode("utf-8", "surrogateescape")] = code
            i += 1
    return out


def check_terminal(deps: Deps) -> List[str]:
    if not deps.isatty():
        return ["NOT-A-TERMINAL: standard input is not a terminal. Only the owner runs this "
                "script, from an interactive shell (`ssh -t`)."]
    return []


def check_agent_env(deps: Deps) -> List[str]:
    hits = sorted(k for k in deps.env
                  if k in AGENT_ENV_EXACT or k.startswith(AGENT_ENV_PREFIXES))
    if hits:
        return [f"AGENT-ENVIRONMENT: this shell carries agent variables ({', '.join(hits)}). "
                f"No agent runs this script."]
    return []


def check_checkout(git: Git) -> List[str]:
    probs = []
    try:
        branch = git.text("symbolic-ref", "--quiet", "--short", "HEAD")
    except GitError:
        branch = "(detached HEAD)"
    if branch != BRANCH:
        probs.append(f"BRANCH: the current branch is {branch}, not {BRANCH}")
    gd = Path(git.text("rev-parse", "--absolute-git-dir")).resolve()
    common = Path(git.text("rev-parse", "--git-common-dir"))
    if not common.is_absolute():
        common = git.repo / common
    if gd != common.resolve():
        probs.append(f"LINKED-WORKTREE: {git.repo} is a linked worktree (git dir {gd}), not the "
                     f"main checkout")
    top = Path(git.text("rev-parse", "--show-toplevel")).resolve()
    if top != git.repo.resolve():
        probs.append(f"NOT-TOPLEVEL: run from the repository root {top}, not {git.repo}")
    return probs


def classify_process(argv: List[str]) -> Optional[str]:
    """Which watched program this argv runs, or None. Looks at the executable and, for a
    Python interpreter, the script or `-m` module it runs. Arguments of other programs (an
    editor open on watcher_agent.py, a shell's -c text) are not matched."""
    def label(name: str) -> Optional[str]:
        if not name or any(c.isspace() for c in name):
            return None
        b = os.path.basename(name)
        if b == "beta_relay.py":
            return "beta_relay.py (a relay)"
        if b == "watcher_agent.py":
            return "watcher_agent.py (WATCHER)"
        if b.startswith("window_optimizer") and b.endswith(".py"):
            return f"{b} (window optimizer)"
        if "coordinator" in b and b.endswith(".py"):
            return f"{b} (coordinator)"
        return None

    if not argv:
        return None
    hit = label(argv[0])
    if hit or not os.path.basename(argv[0]).startswith("python"):
        return hit
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == "-m" and i + 1 < len(argv):
            return label(argv[i + 1].rsplit(".", 1)[-1] + ".py")
        if a == "-c":
            return None
        if a in ("-W", "-X"):
            i += 2
            continue
        if a.startswith("-"):
            i += 1
            continue
        return label(a)
    return None


def check_processes(deps: Deps) -> List[str]:
    probs = []
    for pid, argv in deps.processes():
        if pid == deps.self_pid:
            continue
        what = classify_process(argv)
        if what:
            probs.append(f"PROCESS-RUNNING: pid {pid} is running {what}: {' '.join(argv)[:160]}")
    return probs


def check_dirty(plan: Plan, status: Dict[str, str]) -> List[str]:
    planned = set(plan.paths)
    extra = sorted(p for p in status if p not in planned)
    return [f"DIRTY-PATH: `git status` shows {status[p]!r} {p}, which is not in the plan"
            for p in extra]


def governed_files(repo: Path) -> Tuple[Optional[Tuple[str, ...]], Optional[str]]:
    src = repo / PARITY_GATE_REL
    try:
        tree = ast.parse(src.read_text())
    except (OSError, SyntaxError) as e:
        return None, f"GOVERNED: cannot read GOVERNED_FILES from {PARITY_GATE_REL}: {e}"
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "GOVERNED_FILES" for t in node.targets):
            try:
                return tuple(ast.literal_eval(node.value)), None
            except ValueError as e:
                return None, f"GOVERNED: GOVERNED_FILES in {PARITY_GATE_REL} is not a literal: {e}"
    return None, f"GOVERNED: no GOVERNED_FILES assignment in {PARITY_GATE_REL}"


def check_governed(plan: Plan, repo: Path) -> List[str]:
    gov, err = governed_files(repo)
    if err:
        return [err + " (refusing: the hand-landed set cannot be checked)"]
    probs = []
    for d in plan.paths:
        why = None
        if d in gov:
            why = f"in GOVERNED_FILES ({PARITY_GATE_REL})"
        elif d in HAND_LANDED_FILES:
            why = "on the hand-landed list"
        elif any(d.startswith(p) for p in HAND_LANDED_DIRS):
            why = f"under {next(p for p in HAND_LANDED_DIRS if d.startswith(p))}"
        if why:
            probs.append(f"GOVERNED: {d} is {why}; the owner lands it by hand")
    return probs


def check_entries(plan: Plan, git: Git) -> List[str]:
    probs = []
    repo = git.repo
    seen = set()
    for e in plan.entries:
        if e.dest in seen:
            probs.append(f"DUPLICATE: {e.dest} appears twice in the plan")
        seen.add(e.dest)
        dst = repo / e.dest
        tracked = git.head_blob(e.dest) is not None
        if e.kind == "new":
            if e.source is not None and (dst.exists() or dst.is_symlink()):
                probs.append(f"NEW-EXISTS: {e.dest} is marked new but already exists in the tree")
            if tracked:
                probs.append(f"NEW-EXISTS: {e.dest} is marked new but is tracked at HEAD")
        else:
            if not dst.is_file():
                probs.append(f"MODIFY-MISSING: {e.dest} is marked modify but is not in the tree")
            elif not tracked:
                probs.append(f"MODIFY-MISSING: {e.dest} is marked modify but is not tracked at HEAD")
        src = e.source if e.source is not None else dst
        if e.source is not None and (src.is_symlink() or not src.is_file()):
            probs.append(f"SOURCE-MISSING: {src} is not a regular file")
        elif e.sha is None:
            probs.append(f"SOURCE-MISSING: cannot read {src}")
        if e.expected_sha is not None and e.sha is not None and e.sha != e.expected_sha:
            probs.append(f"SHA-MISMATCH: {src} has sha256 {e.sha}, the manifest says "
                         f"{e.expected_sha}")
        if e.kind == "modify" and e.sha is not None:
            head = git.head_blob(e.dest)
            if head is not None and hashlib.sha256(head).hexdigest() == e.sha:
                probs.append(f"UNCHANGED: {e.dest} would land the same bytes as HEAD")
    if not plan.entries:
        probs.append("EMPTY-PLAN: nothing to land")
    return probs


def check_message(plan: Plan) -> List[str]:
    if not plan.message.strip():
        return ["MESSAGE: the commit message is empty"]
    if "\n" in plan.message or "\r" in plan.message:
        return ["MESSAGE: the commit message must be one line"]
    return []


def check_remotes(git: Git) -> List[str]:
    have = set(git.text("remote").split())
    return [f"REMOTE: remote `{r}` is not configured" for r in REMOTES if r not in have]


def check_self_identity(deps: Deps, git: Git) -> List[str]:
    if deps.self_path is None:
        return []
    here = deps.self_path.resolve()
    want = (git.repo / SELF_REL).resolve()
    if here != want:
        return [f"SELF: this copy is {here}; run the committed one at {want}"]
    head = git.head_blob(SELF_REL)
    if head is None or hashlib.sha256(head).hexdigest() != sha256_file(here):
        return [f"SELF: {SELF_REL} differs from HEAD; only the committed script lands"]
    return []


def all_checks(plan: Plan, git: Git, deps: Deps) -> List[str]:
    probs = list(plan.problems)
    probs += check_terminal(deps)
    probs += check_agent_env(deps)
    probs += check_checkout(git)
    probs += check_self_identity(deps, git)
    probs += check_processes(deps)
    probs += check_dirty(plan, porcelain(git))
    probs += check_governed(plan, git.repo)
    probs += check_entries(plan, git)
    probs += check_message(plan)
    probs += check_remotes(git)
    return probs


# ---------------------------------------------------------------------------------------------
# Landing
# ---------------------------------------------------------------------------------------------
def print_plan(plan: Plan, repo: Path, out) -> None:
    out(f"owner_land: mode={plan.mode}  repo={repo}")
    out(f"  message: {plan.message!r}")
    for e in plan.entries:
        src = f"  <- {e.source}" if e.source is not None else ""
        out(f"  {e.kind:<6}  {e.sha or 'UNREADABLE':<64}  {e.dest}{src}")
    out(f"  {len(plan.entries)} file(s), one commit on {BRANCH}, pushed to "
        f"{' then '.join(REMOTES)}")


def finish_cmd(repo: Path, remotes: Sequence[str]) -> str:
    return f"cd {repo} && " + " && ".join(f"git push {r} {BRANCH}" for r in remotes)


def land(plan: Plan, git: Git, deps: Deps) -> int:
    out = deps.out
    repo = git.repo
    print_plan(plan, repo, out)

    probs = all_checks(plan, git, deps)
    if probs:
        for p in probs:
            out(f"REFUSED  {p}")
        out(f"owner_land: refused ({len(probs)} reason(s)); nothing was changed")
        return EXIT_REFUSED

    try:
        answer = deps.ask(f"Type `{CONFIRM_WORD}` to commit and push these "
                          f"{len(plan.entries)} file(s): ")
    except EOFError:
        answer = ""
    if answer.strip() != CONFIRM_WORD:
        out("owner_land: aborted at the confirmation; nothing was changed")
        return EXIT_ABORTED

    probs = all_checks(plan, git, deps)               # the tree may have moved meanwhile
    if probs:
        for p in probs:
            out(f"REFUSED  {p}")
        out("owner_land: the state changed after the plan was printed; nothing was changed")
        return EXIT_REFUSED

    head_before = git.text("rev-parse", "HEAD")
    prior: Dict[str, Optional[Tuple[bytes, int]]] = {}
    made_dirs: List[Path] = []
    try:
        for e in plan.entries:
            if e.source is None:
                continue
            dst = repo / e.dest
            prior[e.dest] = (dst.read_bytes(), dst.stat().st_mode) if dst.exists() else None
            missing = []
            p = dst.parent
            while not p.exists():
                missing.append(p)
                p = p.parent
            for d in reversed(missing):
                d.mkdir()
                made_dirs.append(d)
            shutil.copyfile(e.source, dst)
            if prior[e.dest] is None:
                shutil.copymode(e.source, dst)
            if sha256_file(dst) != e.sha:
                raise RuntimeError(f"{e.dest}: the copied bytes do not match the plan's sha256")
        for path in plan.paths:
            git.run("add", "--", path)
        staged = set(git.text("diff", "--cached", "--name-only").splitlines())
        if staged != set(plan.paths):
            raise RuntimeError(f"the index holds {sorted(staged)}, the plan is {plan.paths}")
        git.run("commit", "-q", "-m", plan.message)
    except Exception as exc:                           # noqa: BLE001 — every failure rolls back
        if git.text("rev-parse", "HEAD") != head_before:
            out(f"owner_land: FAILED after the commit was created ({exc}); not rolling back")
            return EXIT_PARTIAL
        out(f"owner_land: FAILED before the commit: {exc}")
        git.run("reset", "-q", "--", *plan.paths, check=False)
        for rel, was in prior.items():
            dst = repo / rel
            if was is None:
                if dst.exists():
                    dst.unlink()
            else:
                dst.write_bytes(was[0])
                os.chmod(dst, was[1])
        for d in reversed(made_dirs):
            try:
                d.rmdir()
            except OSError:
                pass
        restored = ", ".join(f"{r} ({'removed' if w is None else 'restored'})"
                             for r, w in prior.items()) or "no files were copied"
        out(f"owner_land: rolled back — unstaged {len(plan.paths)} path(s); {restored}")
        return EXIT_ROLLED_BACK

    commit = git.text("rev-parse", "--short", "HEAD")
    out(f"owner_land: committed {commit}")
    for i, r in enumerate(REMOTES):
        cp = git.run("push", r, BRANCH, check=False)
        if cp.returncode != 0:
            out(f"owner_land: `git push {r} {BRANCH}` FAILED (exit {cp.returncode}): "
                f"{cp.stderr.decode('utf-8', 'replace').strip()}")
            done = ", ".join(REMOTES[:i]) or "no remote"
            out(f"owner_land: commit {commit} is kept locally and is on {done}. Finish with:")
            out(f"  {finish_cmd(repo, REMOTES[i:])}")
            return EXIT_PARTIAL
        out(f"owner_land: pushed to {r}")

    out("git status --porcelain:")
    out(git.run("status", "--porcelain").stdout.decode("utf-8", "replace").rstrip() or "  (clean)")
    out("git log --oneline -1:")
    out("  " + git.text("log", "--oneline", "-1"))
    return EXIT_LANDED


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------
def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="owner_land.py",
        description="The owner's landing script. Only Michael runs it; no agent ever does.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--manifest", nargs="?", const="", metavar="PATH",
                   help=f"land the files a manifest lists (default <inbox>/{MANIFEST_NAME})")
    g.add_argument("--worktree", nargs="+", metavar=("NAME", "PATH"),
                   help="land PATHs from ~/tfm_worktrees/NAME/ (needs -m)")
    g.add_argument("--exchange", metavar="THREAD", help="land a relay round for THREAD")
    ap.add_argument("--inbox", metavar="DIR", help=f"inbox directory (default {INBOX_DEFAULT})")
    ap.add_argument("-m", "--message", help="commit message (worktree mode only)")
    ns = ap.parse_args(argv)
    if ns.worktree is not None:
        if len(ns.worktree) < 2:
            ap.error("--worktree needs a NAME and at least one PATH")
        if ns.message is None:
            ap.error("--worktree needs -m MESSAGE")
    elif ns.message is not None:
        ap.error("-m is for --worktree only; the other modes take the message from their input")
    if ns.inbox is not None and ns.manifest is None:
        ap.error("--inbox is for --manifest only")
    return ns


def run(argv: Sequence[str], repo: Path, deps: Deps) -> int:
    ns = parse_args(argv)
    git = Git(repo, deps)
    try:
        return _run(ns, git, deps)
    except GitError as exc:            # e.g. not a git repository; nothing has been changed yet
        deps.out(f"REFUSED  GIT: {exc}")
        return EXIT_REFUSED


def _run(ns: argparse.Namespace, git: Git, deps: Deps) -> int:
    if ns.manifest is not None:
        inbox = Path(ns.inbox) if ns.inbox else deps.inbox
        manifest = Path(ns.manifest) if ns.manifest else inbox / MANIFEST_NAME
        plan = build_manifest_plan(git, manifest)
    elif ns.worktree is not None:
        plan = build_worktree_plan(git, deps.worktrees_base, ns.worktree[0], ns.worktree[1:],
                                   ns.message)
    else:
        plan = build_exchange_plan(git, ns.exchange, porcelain(git))
    return land(plan, git, deps)


def main(argv: Optional[Sequence[str]] = None) -> int:
    deps = Deps(self_path=Path(__file__))
    return run(sys.argv[1:] if argv is None else argv, Path.cwd(), deps)


if __name__ == "__main__":
    sys.exit(main())
