#!/usr/bin/env python3
"""beta_relay.py — hand one review request to Team Beta (Codex) and check what came back.

Usage:
    python3 scripts/beta_relay.py --request docs/exchange/to_beta/X-NNNN_R<r>_ALPHA_<TYPE>_<topic>.md
                                  [--timeout-min 45]

One run: validate the request, refuse if the reply already exists, snapshot the repository
outside docs/exchange/from_beta/ (HEAD, refs, tracked, untracked and git-ignored files, minus
IGNORED_EXCLUSIONS), the live halt file and the live miner ledger, run Codex non-interactively
with a fixed prompt, snapshot again, check the reply's positions table, append one row to
docs/exchange/LEDGER.md, print the result. Coverage limits: docs/exchange/RELAY_PROTOCOL.md §7.

Exit codes / verdicts:
    0  REVIEW RECEIVED     reply present and well-formed; no change detected in the snapshot
    2  UNAVAILABLE         Codex failed to start / sign in / exited non-zero / timed out / no reply
    3  BOUNDARY VIOLATION  a change detected outside from_beta/ (Beta OR a concurrent writer)
    4  REFUSED             bad request path or the reply already exists; Codex not run
    5  MALFORMED REVIEW    reply exists but has no valid positions table, or an OBJECTION lacks LIVE|THEORETICAL

The relay never reverts, edits, moves or deletes anything. It only reads the request.
Rules: docs/exchange/RELAY_PROTOCOL.md.
"""
from __future__ import annotations

import argparse
import ast
import datetime as _dt
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------------------------
EXIT_RECEIVED = 0
EXIT_UNAVAILABLE = 2
EXIT_VIOLATION = 3
EXIT_REFUSED = 4
EXIT_MALFORMED = 5

VERDICTS = {
    EXIT_RECEIVED: "REVIEW RECEIVED",
    EXIT_UNAVAILABLE: "UNAVAILABLE",
    EXIT_VIOLATION: "BOUNDARY VIOLATION",
    EXIT_REFUSED: "REFUSED",
    EXIT_MALFORMED: "MALFORMED REVIEW",
}

POSITION_WORDS = ("NO OBJECTION", "OBJECTION", "SUGGESTION", "NEEDS EVIDENCE")

TO_BETA_REL = "docs/exchange/to_beta"
FROM_BETA_REL = "docs/exchange/from_beta"
LEDGER_REL = "docs/exchange/LEDGER.md"
DEFERRED_REL = "docs/exchange/DEFERRED.md"
LOG_DIR_REL = "logs/beta_relay"

# Committed sources the live paths are read from (no hardcoded live paths).
HALT_SOURCE_REL = "agents/safety/kill_switch.py"          # KillSwitch.halt_file_path
STAGING_MANIFEST_REL = "agent_manifests/window_optimizer.json"  # default_params.staging_dir
LEDGER_DB_NAME = "miner_ledger.db"

REQUEST_RE = re.compile(r"^X-(?P<id>[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*)_R(?P<round>[1-9][0-9]*)_ALPHA_.+\.md$")

DEFAULT_TIMEOUT_MIN = 45.0
KILL_GRACE_S = 10.0

# Git-ignored files in the repository are snapshotted too. Files up to this size are hashed;
# larger ones are recorded by size and mtime (ns) only.
IGNORED_HASH_MAX_BYTES = 8 * 1024 * 1024
# The only ignored paths left out of the snapshot. Printed on every run.
IGNORED_EXCLUSIONS = (
    "any `__pycache__/` directory",
    "any `*.pyc` file",
    "any `.pytest_cache/` directory",
    "`logs/beta_relay/` (the relay's own logs)",
)
REASONING_EFFORT = "high"

LEDGER_HEADER = (
    "# Alpha–Beta relay ledger\n"
    "\n"
    "Written only by `scripts/beta_relay.py`, one row per relay run that launched Codex "
    "(a `REFUSED` run launches nothing and writes no row). Rules: "
    "`docs/exchange/RELAY_PROTOCOL.md`.\n"
    "\n"
    "| Thread | Round | Date | Request sha256 | Reply sha256 | Positions | Verdict | Other from_beta changes |\n"
    "|---|---|---|---|---|---|---|---|\n"
)

DEFERRED_HEADER = (
    "# Deferred THEORETICAL objections\n"
    "\n"
    "Written only by `scripts/beta_relay.py`. One row per `OBJECTION` that Beta labelled "
    "`THEORETICAL`: it needs conditions that do not exist on this system today, so it holds no "
    "commit (owner ruling, 2026-11-12; `AGENTS.md` §5). Rules: `docs/exchange/RELAY_PROTOCOL.md`.\n"
    "\n"
    "| Thread | Round | Question | Date | Beta's reason |\n"
    "|---|---|---|---|---|\n"
)


# ---------------------------------------------------------------------------------------------
# The prompt. Built here and only here. Nothing else is ever appended.
# ---------------------------------------------------------------------------------------------
def build_prompt(request_rel: str) -> str:
    return f"Read {request_rel} and reply as it asks."


# ---------------------------------------------------------------------------------------------
# Launching
# ---------------------------------------------------------------------------------------------
# A launcher takes (prompt, repo_root, log_file_handle) and returns a started subprocess.Popen.
# It must start the child in its own session (use spawn()) so a timeout can kill the whole
# process group. Tests inject a fake; the default runs the real Codex.
Launcher = Callable[[str, Path, "object"], subprocess.Popen]


def spawn(argv: List[str], cwd: Path, log_fh) -> subprocess.Popen:
    return subprocess.Popen(
        argv, cwd=str(cwd), stdin=subprocess.DEVNULL, stdout=log_fh, stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def codex_argv(prompt: str, repo: Path) -> List[str]:
    # Flags settled in docs/CLAUDE_CODE_REPORT_BETA_RELAY.md §1:
    #   -s workspace-write              least sandbox that can write from_beta/ (read-only cannot)
    #   -c approval_policy="never"      no approval prompt can ever block a headless run
    #   -c model_reasoning_effort="high"  Beta's reasoning effort (log header: `reasoning effort: high`)
    #   --strict-config                 an unknown -c key is a hard error, never silently ignored
    #   -C <repo>                       working root = repository root (loads AGENTS.md)
    #   --color never                   plain log
    # The prompt is the single positional argument; stdin is /dev/null so nothing is appended.
    return [
        "codex", "exec",
        "--strict-config",
        "-s", "workspace-write",
        "-c", 'approval_policy="never"',
        "-c", f'model_reasoning_effort="{REASONING_EFFORT}"',
        "-C", str(repo),
        "--color", "never",
        prompt,
    ]


def codex_launcher(prompt: str, repo: Path, log_fh) -> subprocess.Popen:
    argv = codex_argv(prompt, repo)
    log_fh.write(("argv: " + json.dumps(argv) + "\n").encode())
    log_fh.flush()
    return spawn(argv, repo, log_fh)


def _kill_group(proc: subprocess.Popen) -> None:
    for sig, wait_s in ((signal.SIGTERM, KILL_GRACE_S), (signal.SIGKILL, KILL_GRACE_S)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass
        except PermissionError:
            proc.send_signal(sig)
        try:
            proc.wait(timeout=wait_s)
            return
        except subprocess.TimeoutExpired:
            continue


# ---------------------------------------------------------------------------------------------
# Live-path resolution (from committed sources)
# ---------------------------------------------------------------------------------------------
def resolve_halt_path(repo: Path) -> Path:
    """KillSwitch.halt_file_path default, read by AST (no import side effects)."""
    tree = ast.parse((repo / HALT_SOURCE_REL).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "KillSwitch":
            for stmt in node.body:
                if (isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
                        and stmt.target.id == "halt_file_path" and isinstance(stmt.value, ast.Constant)):
                    return Path(stmt.value.value)
    raise RuntimeError(f"KillSwitch.halt_file_path not found in {HALT_SOURCE_REL}")


def resolve_ledger_dir(repo: Path) -> Path:
    d = json.loads((repo / STAGING_MANIFEST_REL).read_text())
    return Path(os.path.expanduser(d["default_params"]["staging_dir"]))


# ---------------------------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------------------------
def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=str(repo), capture_output=True, check=True).stdout


_EXCL = (".", f":(exclude){FROM_BETA_REL}")


@dataclass
class Snapshot:
    head: str
    refs: str
    porcelain: List[str]
    diff_sha: str
    tracked_dirty: Dict[str, str]
    untracked: Dict[str, str]
    halt: Dict[str, bool]
    ledger: Dict[str, Optional[str]]
    from_beta: Dict[str, str] = field(default_factory=dict)
    ignored: Dict[str, str] = field(default_factory=dict)


def ignored_is_excluded(rel: str) -> bool:
    """True for the ignored paths in IGNORED_EXCLUSIONS (rel is repo-relative, POSIX)."""
    parts = rel.split("/")
    return ("__pycache__" in parts[:-1] or rel.endswith(".pyc") or ".pytest_cache" in parts[:-1]
            or rel.startswith("logs/beta_relay/"))


def _ignored_record(p: Path) -> str:
    if p.is_symlink():
        return f"symlink->{os.readlink(p)}"
    if not p.is_file():
        return "<not-a-file>"
    st = p.stat()
    if st.st_size <= IGNORED_HASH_MAX_BYTES:
        return f"sha256={_sha_file(p)}"
    return f"size={st.st_size} mtime_ns={st.st_mtime_ns}"


def take_snapshot(repo: Path, halt_paths: List[Path], ledger_dir: Path) -> Snapshot:
    head = _git(repo, "rev-parse", "HEAD").decode().strip()
    refs = hashlib.sha256(_git(repo, "for-each-ref", "--format=%(refname) %(objectname)")).hexdigest()
    porcelain = [x for x in _git(repo, "status", "--porcelain=v1", "-z", "--untracked-files=all",
                                 "--", *_EXCL).decode("utf-8", "surrogateescape").split("\0") if x]
    diff_sha = hashlib.sha256(_git(repo, "diff", "HEAD", "--binary", "--", *_EXCL)).hexdigest()

    tracked_dirty: Dict[str, str] = {}
    for rel in _git(repo, "diff", "HEAD", "--name-only", "-z", "--", *_EXCL).decode(
            "utf-8", "surrogateescape").split("\0"):
        if rel:
            p = repo / rel
            tracked_dirty[rel] = _sha_file(p) if p.is_file() else "<absent>"

    untracked: Dict[str, str] = {}
    for rel in _git(repo, "ls-files", "--others", "--exclude-standard", "-z", "--", *_EXCL).decode(
            "utf-8", "surrogateescape").split("\0"):
        if rel:
            p = repo / rel
            untracked[rel] = _sha_file(p) if p.is_file() else "<not-a-file>"

    ignored: Dict[str, str] = {}
    for rel in _git(repo, "ls-files", "--others", "--ignored", "--exclude-standard", "-z",
                    "--", *_EXCL).decode("utf-8", "surrogateescape").split("\0"):
        if rel and not ignored_is_excluded(rel):
            ignored[rel] = _ignored_record(repo / rel)

    halt = {str(p): p.exists() for p in halt_paths}

    # Live ledger: plain-byte sha256 of the db, and every -wal/-shm sidecar's size and sha256.
    # Never opened with SQLite.
    ledger: Dict[str, Optional[str]] = {}
    db = ledger_dir / LEDGER_DB_NAME
    ledger[LEDGER_DB_NAME] = _sha_file(db) if db.is_file() else None
    for suffix in ("-wal", "-shm", "-journal"):
        sc = ledger_dir / (LEDGER_DB_NAME + suffix)
        ledger[sc.name] = f"size={sc.stat().st_size} sha256={_sha_file(sc)}" if sc.is_file() else None

    fb_dir = repo / FROM_BETA_REL
    from_beta: Dict[str, str] = {}
    if fb_dir.is_dir():
        for p in sorted(fb_dir.rglob("*")):
            if p.is_file():
                from_beta[p.relative_to(repo).as_posix()] = _sha_file(p)

    return Snapshot(head, refs, porcelain, diff_sha, tracked_dirty, untracked, halt, ledger, from_beta,
                    ignored)


def compare_snapshots(a: Snapshot, b: Snapshot) -> List[str]:
    """Every difference outside from_beta/, as human-readable lines. Empty list = no change."""
    out: List[str] = []
    if a.head != b.head:
        out.append(f"HEAD moved: {a.head} -> {b.head}")
    if a.refs != b.refs:
        out.append("git refs changed (branch, tag or other ref created, moved or deleted)")
    for line in sorted(set(a.porcelain) ^ set(b.porcelain)):
        out.append(f"git status line {'gone' if line in a.porcelain else 'new'}: {line!r}")
    for rel in sorted(set(a.tracked_dirty) | set(b.tracked_dirty)):
        if a.tracked_dirty.get(rel) != b.tracked_dirty.get(rel):
            out.append(f"tracked file changed: {rel}")
    if a.diff_sha != b.diff_sha and not any(x.startswith("tracked file changed") for x in out):
        out.append("sha256 of `git diff HEAD` changed (index/mode change)")
    for rel in sorted(set(a.untracked) | set(b.untracked)):
        x, y = a.untracked.get(rel), b.untracked.get(rel)
        if x != y:
            kind = "created" if x is None else "deleted" if y is None else "modified"
            out.append(f"untracked file {kind}: {rel}")
    for rel in sorted(set(a.ignored) | set(b.ignored)):
        x, y = a.ignored.get(rel), b.ignored.get(rel)
        if x != y:
            kind = "created" if x is None else "deleted" if y is None else "modified"
            out.append(f"ignored file {kind}: {rel}")
    for p in sorted(set(a.halt) | set(b.halt)):
        if a.halt.get(p) != b.halt.get(p):
            out.append(f"halt file {p}: exists {a.halt.get(p)} -> {b.halt.get(p)}")
    for n in sorted(set(a.ledger) | set(b.ledger)):
        if a.ledger.get(n) != b.ledger.get(n):
            out.append(f"live ledger {n}: {a.ledger.get(n)} -> {b.ledger.get(n)}")
    return out


def from_beta_changes(a: Snapshot, b: Snapshot, expected_rel: str) -> List[str]:
    out = []
    for rel in sorted(set(a.from_beta) | set(b.from_beta)):
        if rel == expected_rel:
            continue
        x, y = a.from_beta.get(rel), b.from_beta.get(rel)
        if x != y:
            kind = "new" if x is None else "deleted" if y is None else "modified"
            out.append(f"{kind}: {rel}")
    return out


# ---------------------------------------------------------------------------------------------
# Reply check
# ---------------------------------------------------------------------------------------------
def _cells(line: str) -> List[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _plain(cell: str) -> str:
    c = cell.strip()
    for wrap in ("**", "__", "`", "*", "_"):
        if len(c) > 2 * len(wrap) and c.startswith(wrap) and c.endswith(wrap):
            c = c[len(wrap):-len(wrap)].strip()
    return c


_SEP_RE = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")


# Severity cell on an OBJECTION row (owner ruling 2026-11-12; AGENTS.md §5):
#   LIVE — <one sentence>   or   THEORETICAL — <one sentence>
SEVERITY_WORDS = ("LIVE", "THEORETICAL")
_SEVERITY_RE = re.compile(r"^[*_`]*(LIVE|THEORETICAL)[*_`]*\s*(?:—|–|-|:)\s*(\S.*)$")

# A positions row: (question, position, severity or "", reason or "").
Row = Tuple[str, str, str, str]


def parse_positions(text: str) -> Tuple[Optional[List[Row]], str]:
    """Find the first markdown table with a 'Position' column. Returns (rows, error).
    Every OBJECTION row must carry a Severity cell `LIVE — reason` or `THEORETICAL — reason`."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if not line.strip().startswith("|"):
            continue
        header = [_plain(c).lower() for c in _cells(line)]
        if "position" not in header:
            continue
        if i + 1 >= len(lines) or not _SEP_RE.match(lines[i + 1].strip()):
            continue
        col = header.index("position")
        sev_col = header.index("severity") if "severity" in header else None
        rows: List[Row] = []
        for row in lines[i + 2:]:
            if not row.strip().startswith("|"):
                break
            cells = _cells(row)
            if col >= len(cells):
                return None, f"positions row has no position cell: {row.strip()!r}"
            q = _plain(cells[0]) if col != 0 else f"row{len(rows) + 1}"
            pos = _plain(cells[col])
            if pos not in POSITION_WORDS:
                return None, f"position cell {pos!r} (row {q!r}) is not one of {list(POSITION_WORDS)}"
            sev_cell = cells[sev_col].strip() if sev_col is not None and sev_col < len(cells) else ""
            m = _SEVERITY_RE.match(sev_cell)
            sev, reason = (m.group(1), m.group(2).strip()) if m else ("", "")
            sev_ok = bool(m)
            if pos == "OBJECTION" and not sev_ok:
                return None, (f"OBJECTION row {q!r} has no severity: the Severity cell must be "
                              f"'LIVE — <reason>' or 'THEORETICAL — <reason>', got {sev_cell!r}")
            if pos != "OBJECTION":
                sev, reason = "", ""
            rows.append((q, pos, sev, reason))
        if not rows:
            return None, "positions table has no rows"
        return rows, ""
    return None, "no positions table (no markdown table with a 'Position' column)"


# ---------------------------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------------------------
def _md(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


def append_ledger(repo: Path, thread: str, rnd: str, req_sha: str, reply_sha: str,
                  positions: str, verdict: str, extras: List[str]) -> None:
    path = repo / LEDGER_REL
    if not path.exists():
        path.write_text(LEDGER_HEADER)
    row = "| " + " | ".join(_md(c) for c in (
        thread, f"R{rnd}", _dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"),
        f"`{req_sha}`", f"`{reply_sha}`" if reply_sha != "—" else "—",
        positions or "—", verdict, "; ".join(extras) if extras else "—")) + " |\n"
    with open(path, "a") as f:
        f.write(row)


def append_deferred(repo: Path, thread: str, rnd: str, rows: List[Row]) -> List[str]:
    """One DEFERRED.md row per THEORETICAL objection. Returns the questions appended."""
    theoretical = [r for r in rows if r[1] == "OBJECTION" and r[2] == "THEORETICAL"]
    if not theoretical:
        return []
    path = repo / DEFERRED_REL
    if not path.exists():
        path.write_text(DEFERRED_HEADER)
    date = _dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    with open(path, "a") as f:
        for q, _pos, _sev, reason in theoretical:
            f.write("| " + " | ".join(_md(c) for c in (thread, f"R{rnd}", q, date, reason)) + " |\n")
    return [r[0] for r in theoretical]


# ---------------------------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------------------------
@dataclass
class Result:
    code: int
    verdict: str
    reply_rel: str = ""
    positions: Optional[List[Row]] = None
    violations: List[str] = field(default_factory=list)
    extras: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    log_rel: str = ""


def _refuse(msg: str) -> Result:
    return Result(EXIT_REFUSED, VERDICTS[EXIT_REFUSED], notes=[msg])


def run_relay(request: str, repo: Path, launcher: Launcher = codex_launcher,
              timeout_s: float = DEFAULT_TIMEOUT_MIN * 60,
              halt_paths: Optional[List[Path]] = None, ledger_dir: Optional[Path] = None) -> Result:
    repo = repo.resolve()
    to_beta = (repo / TO_BETA_REL).resolve()

    # 1. Validate the request.
    raw = Path(request)
    req = (raw if raw.is_absolute() else Path.cwd() / raw)
    if not req.exists() and not raw.is_absolute():
        req = repo / raw
    if req.is_symlink():
        return _refuse(f"request is a symlink: {request}")
    req = req.resolve()
    if req.parent != to_beta:
        return _refuse(f"request is not directly inside {TO_BETA_REL}/: {request}")
    if not req.is_file():
        return _refuse(f"request does not exist: {request}")
    m = REQUEST_RE.match(req.name)
    if not m:
        return _refuse(f"request name does not match X-<id>_R<round>_ALPHA_<...>.md: {req.name}")
    thread, rnd = "X-" + m.group("id"), m.group("round")
    request_rel = req.relative_to(repo).as_posix()

    # 2. Expected reply; never overwritten.
    reply_rel = f"{FROM_BETA_REL}/{thread}_R{rnd}_BETA_REVIEW.md"
    reply = repo / reply_rel
    if reply.exists() or reply.is_symlink():
        return Result(EXIT_REFUSED, VERDICTS[EXIT_REFUSED], reply_rel=reply_rel,
                      notes=[f"reply already exists: {reply_rel}; a review is never overwritten"])

    halt_paths = halt_paths if halt_paths is not None else [resolve_halt_path(repo)]
    ledger_dir = ledger_dir if ledger_dir is not None else resolve_ledger_dir(repo)
    req_sha = _sha_file(req)

    # 3. Snapshot.
    before = take_snapshot(repo, halt_paths, ledger_dir)

    # 4. Run Codex.
    log_dir = repo / LOG_DIR_REL
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{thread}_R{rnd}.log"
    prompt = build_prompt(request_rel)
    notes: List[str] = []
    launch_failed = False
    t0 = time.monotonic()
    with open(log_path, "ab") as log_fh:
        log_fh.write(f"\n===== relay run {_dt.datetime.now().astimezone().isoformat()} "
                     f"request={request_rel} timeout_s={timeout_s}\n".encode())
        log_fh.write(("prompt: " + json.dumps(prompt) + "\n").encode())
        log_fh.write(("ignored-file exclusions: " + json.dumps(list(IGNORED_EXCLUSIONS)) + "\n").encode())
        log_fh.flush()
        try:
            proc = launcher(prompt, repo, log_fh)
        except Exception as e:  # failed to start
            launch_failed = True
            notes.append(f"Codex failed to start: {type(e).__name__}: {e}")
            proc = None
        if proc is not None:
            try:
                rc = proc.wait(timeout=timeout_s)
                if rc != 0:
                    launch_failed = True
                    notes.append(f"Codex exited with status {rc}")
            except subprocess.TimeoutExpired:
                _kill_group(proc)
                launch_failed = True
                notes.append(f"Codex timed out after {timeout_s:.0f}s; process group terminated "
                             f"(final status {proc.returncode})")
        elapsed = time.monotonic() - t0
        log_fh.write(f"===== end elapsed_s={elapsed:.1f} notes={json.dumps(notes)}\n".encode())

    # 5. Snapshot again and compare.
    after = take_snapshot(repo, halt_paths, ledger_dir)
    violations = compare_snapshots(before, after)
    extras = from_beta_changes(before, after, reply_rel)

    # 6. Check the reply.
    positions = None
    reply_sha = "—"
    malformed = ""
    if reply.is_file():
        reply_sha = _sha_file(reply)
        text = reply.read_text(errors="replace")
        if not text.strip():
            malformed = "reply file is empty"
        else:
            positions, malformed = parse_positions(text)
    elif not launch_failed:
        launch_failed = True
        notes.append(f"no reply file at {reply_rel}")

    if violations:
        code = EXIT_VIOLATION
        notes.append("A boundary violation can also come from another session writing during the run; "
                     "the relay cannot tell the two apart. Nothing was reverted.")
    elif launch_failed:
        code = EXIT_UNAVAILABLE
    elif malformed:
        code = EXIT_MALFORMED
        notes.append(f"malformed: {malformed}. The file is left as Beta wrote it.")
    else:
        code = EXIT_RECEIVED
    if malformed and code != EXIT_MALFORMED:
        notes.append(f"reply also malformed: {malformed}")

    # 7. Ledger.
    pos_str = "; ".join(f"{q} {p}" + (f" ({s})" if s else "") for q, p, s, _ in positions) if positions else ""
    append_ledger(repo, thread, rnd, req_sha, reply_sha, pos_str, VERDICTS[code], extras)
    if positions:
        deferred = append_deferred(repo, thread, rnd, positions)
        if deferred:
            notes.append(f"THEORETICAL objection(s) appended to {DEFERRED_REL}: {', '.join(deferred)}")

    return Result(code, VERDICTS[code], reply_rel, positions, violations, extras, notes,
                  elapsed, log_path.relative_to(repo).as_posix())


def print_exclusions() -> None:
    print("Snapshot covers git-ignored files in the repository except (not checked):", flush=True)
    for e in IGNORED_EXCLUSIONS:
        print(f"  - {e}", flush=True)


def print_result(r: Result) -> None:
    print(f"VERDICT: {r.verdict} (exit {r.code})")
    if r.reply_rel:
        print(f"reply: {r.reply_rel}")
    if r.log_rel:
        print(f"log: {r.log_rel}  elapsed: {r.elapsed_s:.1f}s")
    if r.positions:
        print("| Question | Position | Severity |\n|---|---|---|")
        for q, p, s, why in r.positions:
            print(f"| {q} | {p} | {f'{s} — {why}' if s else '—'} |")
    for v in r.violations:
        print(f"CHANGED OUTSIDE from_beta/: {v}")
    for e in r.extras:
        print(f"other from_beta/ change: {e}")
    for n in r.notes:
        print(f"note: {n}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Alpha→Beta review relay (see docs/exchange/RELAY_PROTOCOL.md)")
    ap.add_argument("--request", required=True, help="docs/exchange/to_beta/X-<id>_R<r>_ALPHA_<...>.md")
    ap.add_argument("--timeout-min", type=float, default=DEFAULT_TIMEOUT_MIN)
    a = ap.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    print_exclusions()
    r = run_relay(a.request, repo, timeout_s=a.timeout_min * 60)
    print_result(r)
    return r.code


if __name__ == "__main__":
    sys.exit(main())
