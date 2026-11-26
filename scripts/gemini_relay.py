#!/usr/bin/env python3
"""gemini_relay.py — hand one review request to Gemini (Google Antigravity CLI) and file the reply.

Usage:
    python3 scripts/gemini_relay.py --request docs/exchange/to_gemini/X-NNNN_R<r>_ALPHA_<TYPE>_<topic>.md
                                    [--timeout-min 45]

Called the same way as scripts/beta_relay.py, with its own folders, so a request can never reach
the wrong reviewer. scripts/beta_relay.py is imported for the shared rules (request name,
positions table, exit codes) and is not changed.

One run: validate the request (name, folder, an `Owner approval:` line), refuse if the reply
already exists, build a review copy of this working tree outside the repository, write the
lockdown settings, run `agy` non-interactively on the copy with a fixed prompt, check the copy
was not changed, check the reply's positions table, write the reply file from Gemini's answer,
append one row to docs/exchange/GEMINI_LEDGER.md, print the result.

Differences from the Beta relay, all from the owner ruling of 2026-11-26 (Gemini is an occasional
second reviewer on items the owner chooses; read-only, locked down; not binding):
  - Gemini never sees the live tree. It reads a copy; the relay, not Gemini, writes the reply.
  - A request must carry the owner's approval for this use.
  - No Gemini position holds a commit. Nothing is written to DEFERRED.md.
  - The live tree is not snapshotted, so other sessions may work while the relay runs.

Exit codes / verdicts (same numbers as beta_relay.py):
    0  REVIEW RECEIVED     reply filed and well-formed; the review copy was not changed
    2  UNAVAILABLE         agy failed to start / exited non-zero / timed out / gave no answer
    3  BOUNDARY VIOLATION  the review copy was changed during the run (Gemini is read-only)
    4  REFUSED             bad request, no owner approval, or the reply already exists; agy not run
    5  MALFORMED REVIEW    answer filed but it has no valid positions table

Rules: docs/exchange/GEMINI_RELAY_PROTOCOL.md.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import beta_relay as _beta  # noqa: E402  shared rules only; nothing in it is modified

# ---------------------------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------------------------
EXIT_RECEIVED = _beta.EXIT_RECEIVED
EXIT_UNAVAILABLE = _beta.EXIT_UNAVAILABLE
EXIT_VIOLATION = _beta.EXIT_VIOLATION
EXIT_REFUSED = _beta.EXIT_REFUSED
EXIT_MALFORMED = _beta.EXIT_MALFORMED
VERDICTS = _beta.VERDICTS
POSITION_WORDS = _beta.POSITION_WORDS
REQUEST_RE = _beta.REQUEST_RE

TO_GEMINI_REL = "docs/exchange/to_gemini"
FROM_GEMINI_REL = "docs/exchange/from_gemini"
LEDGER_REL = "docs/exchange/GEMINI_LEDGER.md"
LOG_DIR_REL = "logs/gemini_relay"

# Where review copies live: outside the repository, outside every protected path, and not /tmp
# (the lockdown denies reading /tmp, and VM101 wipes it at boot).
WORK_ROOT_ENV = "GEMINI_RELAY_WORKDIR"
WORK_ROOT_DEFAULT = "~/gemini_reviews"

# Settled by the trial of 2026-11-26 (claude/GEMINI_ANTIGRAVITY_USAGE_20261126.md):
AGY_BIN = "agy"
GEMINI_MODEL = "gemini-4-hydrogen-xhigh"
AGY_MODE = "plan"            # headless `--mode auto` is not available on this account
DEFAULT_TIMEOUT_MIN = 45.0
KILL_GRACE_S = _beta.KILL_GRACE_S
MAX_FINDINGS = 10

# Other reviewers' replies and the relay ledgers are left out of the copy, so the review is
# independent. Printed on every run.
COPY_EXCLUDED = (
    "docs/exchange/from_beta/",
    "docs/exchange/from_gemini/",
    "docs/exchange/LEDGER.md",
    "docs/exchange/DEFERRED.md",
    "docs/exchange/GEMINI_LEDGER.md",
)

# Paths under the home directory Gemini must not read (the trial's deny list). The repository's
# main checkout, this checkout, SYSTEM_PROTECTED and the relay's own run folder are added at run time.
HOME_PROTECTED = (
    "distributed_prng_analysis",   # the TFM tree (CLAUDE.md §0), also when run from a clone
    ".config", ".ssh", ".claude", ".codex", ".local/state/api-agents",
    "tfm_worktrees", "tfm_inbox", "agy_trial/keys", "agy_trial/setup", "claude-api-agents",
)
# Command rules, word for word from the trial's review lockdown
# (~/agy_trial/setup/trial_settings_review.json.t1b). They match a command as typed: they stop
# the ordinary case and are not a cage. No relative path may appear in any rule: one broke the
# sandbox in the trial.
COMMAND_DENY = (
    "command(git commit)",
    "command(git push)",
    "command(regex:.*git .*(commit|push).*)",
    "command(sudo)",
    "command(ssh)",
    "command(scp)",
    "command(rsync)",
    "command(rm -rf)",
    "command(regex:.*(curl|wget) .*)",
    "command(regex:.*--run-pipeline.*)",
    "command(regex:.*(pytest|unittest|test_ch14|meta_prediction_optimizer|watcher_agent"
    "|prediction_generator|run_step|nohup|pip ).*)",
)
# System paths Gemini must not read. /tmp holds other sessions' scratch files and Beta's.
SYSTEM_PROTECTED = ("/tmp",)
COMMAND_ALLOW = ("command(*)",)   # headless soft-denies anything needing approval otherwise

OWNER_LINE_RE = re.compile(r"^\s*(?:[-*]\s*)?\**\s*Owner approval\s*\**\s*:\s*\**\s*(\S.*?)\s*$",
                           re.IGNORECASE | re.MULTILINE)

LEDGER_HEADER = (
    "# Alpha–Gemini relay ledger\n"
    "\n"
    "Written only by `scripts/gemini_relay.py`, one row per relay run that launched `agy` "
    "(a `REFUSED` run launches nothing and writes no row). Gemini's positions are not binding "
    "and hold nothing (owner ruling, 2026-11-26). Rules: "
    "`docs/exchange/GEMINI_RELAY_PROTOCOL.md`.\n"
    "\n"
    "| Thread | Round | Date | Request sha256 | Reply sha256 | Positions | Verdict | Denied actions "
    "| Tokens | Copy of |\n"
    "|---|---|---|---|---|---|---|---|---|---|\n"
)


# ---------------------------------------------------------------------------------------------
# The prompt. Built here and only here. It carries the owner's five prompt rules, so no session
# can loosen them. Nothing else is ever appended.
# ---------------------------------------------------------------------------------------------
def build_prompt(request_rel: str) -> str:
    return (
        f"You are a second reviewer of this repository copy. Read {request_rel} and answer it.\n"
        "\n"
        "Hard rules:\n"
        "- Do not edit, create or delete any file. Never commit or push, never launch any "
        "pipeline. Do not run tests or any program. You may use shell commands only to read and "
        "search files, and only in the foreground.\n"
        "- Answer only the numbered questions in the request. Do not widen the task.\n"
        "- AGENTS.md in this copy is addressed to another reviewer. Its yardstick, learning-loop "
        "and prior-record sections apply to you. Its statements about who you are and about "
        "writing reply files do not: your reply is your final message, and you write no file. "
        "Other reviewers' replies are deliberately not in this copy.\n"
        "\n"
        "Your final message, in this order:\n"
        "1. A positions table, one row per question, with exactly these columns:\n"
        "   | Question | Position | Severity |\n"
        "   Position is one of: NO OBJECTION, OBJECTION, SUGGESTION, NEEDS EVIDENCE.\n"
        "   Severity is required on every OBJECTION row: `LIVE — <one sentence>` (it would cause "
        "a wrong result in a realistic run on this system as it is today) or `THEORETICAL — <one "
        "sentence>` (it needs conditions that do not exist here today). Put `—` on other rows.\n"
        "2. Your reasoning for each row. Every claim about the code carries a file:line you read "
        "in this session, written as plain text `path:line` relative to the repository root, "
        "with no links and no absolute paths. For each defect give: what goes wrong, one concrete input that "
        "triggers it, and your confidence (high, medium or low). Propose the smallest change "
        "that would fix it, in words; do not write the patch.\n"
        "3. Non-vacuity: for every test, gate or comparison in the work under review, say what "
        "it compares and how many items. Report any check that compares nothing.\n"
        f"4. At most {MAX_FINDINGS} findings in total, most serious first. A known, recorded "
        "item is a status, not a finding.\n"
        "5. A short list titled `Not handled`: what you chose not to look at, and why.\n"
    )


# ---------------------------------------------------------------------------------------------
# Lockdown settings (session only; the owner's own settings.json is never touched)
# ---------------------------------------------------------------------------------------------
def protected_paths(repo: Path, main_checkout: Path, run_dir: Path, home: Path) -> List[Path]:
    paths = [main_checkout, repo] + [home / p for p in HOME_PROTECTED] + [Path(p) for p in SYSTEM_PROTECTED] + [run_dir]
    out: List[Path] = []
    for p in paths:
        if p not in out:
            out.append(p)
    return out


def lockdown_settings(protected: List[Path]) -> dict:
    for p in protected:
        if not p.is_absolute():
            raise ValueError(f"lockdown path is not absolute: {p}")
    deny = [f"read_file({p})" for p in protected] + list(COMMAND_DENY)
    return {"enableTerminalSandbox": True,
            "permissions": {"deny": deny, "allow": list(COMMAND_ALLOW)}}


def _is_under(p: Path, parent: Path) -> bool:
    try:
        p.relative_to(parent)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------------------------
# The review copy: this working tree as it is now (tracked files with their current content,
# plus untracked files git does not ignore), minus COPY_EXCLUDED. No commit is made anywhere.
# ---------------------------------------------------------------------------------------------
def copy_is_excluded(rel: str) -> bool:
    return any(rel == e or (e.endswith("/") and rel.startswith(e)) for e in COPY_EXCLUDED)


@dataclass
class CopyInfo:
    head: str
    files: int
    dirty: int
    skipped: List[str] = field(default_factory=list)


def build_copy(repo: Path, dest: Path) -> CopyInfo:
    head = _beta._git(repo, "rev-parse", "HEAD").decode().strip()
    dirty = len([x for x in _beta._git(repo, "status", "--porcelain=v1", "-z",
                                       "--untracked-files=all").decode(
        "utf-8", "surrogateescape").split("\0") if x])
    listed = _beta._git(repo, "ls-files", "-z", "--cached", "--others", "--exclude-standard").decode(
        "utf-8", "surrogateescape").split("\0")
    info = CopyInfo(head, 0, dirty)
    dest.mkdir(parents=True)
    for rel in sorted(set(x for x in listed if x)):
        if copy_is_excluded(rel):
            continue
        src = repo / rel
        if src.is_symlink():
            info.skipped.append(f"symlink: {rel}")
            continue
        if not src.is_file():
            continue          # deleted in the working tree, or a nested repository entry
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, out)
        info.files += 1
    return info


def digest_tree(root: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        if p.is_symlink():
            out[rel] = f"symlink->{os.readlink(p)}"
        elif p.is_file():
            out[rel] = _beta._sha_file(p)
    return out


def compare_digests(a: Dict[str, str], b: Dict[str, str]) -> List[str]:
    out: List[str] = []
    for rel in sorted(set(a) | set(b)):
        x, y = a.get(rel), b.get(rel)
        if x != y:
            kind = "created" if x is None else "deleted" if y is None else "modified"
            out.append(f"review copy file {kind}: {rel}")
    return out


# ---------------------------------------------------------------------------------------------
# Launching
# ---------------------------------------------------------------------------------------------
# A launcher takes (prompt, copy_dir, settings_path, out_fh, err_fh, print_timeout_min) and
# returns a started subprocess.Popen in its own session. Tests inject a fake; the default runs
# the real agy.
Launcher = Callable[[str, Path, Path, "object", "object", int], subprocess.Popen]


def agy_argv(prompt: str, settings_path: Path, print_timeout_min: int) -> List[str]:
    #   -p <prompt>                 one non-interactive turn
    #   --settings <file>           session-only lockdown; the owner's settings.json is untouched
    #   --mode plan                 read-only planning mode
    #   --output-format stream-json one JSON line per step, so every command is on record
    #   --print-timeout <n>m        agy's own limit, just inside the relay's
    return [
        AGY_BIN, "-p", prompt,
        "--settings", str(settings_path),
        "--mode", AGY_MODE,
        "--model", GEMINI_MODEL,
        "--output-format", "stream-json",
        "--print-timeout", f"{print_timeout_min}m",
    ]


def agy_launcher(prompt: str, copy_dir: Path, settings_path: Path, out_fh, err_fh,
                 print_timeout_min: int) -> subprocess.Popen:
    return subprocess.Popen(
        agy_argv(prompt, settings_path, print_timeout_min), cwd=str(copy_dir),
        stdin=subprocess.DEVNULL, stdout=out_fh, stderr=err_fh, start_new_session=True,
    )


# ---------------------------------------------------------------------------------------------
# Reading agy's output
# ---------------------------------------------------------------------------------------------
def parse_agy_output(text: str) -> Tuple[Optional[dict], List[str]]:
    """Returns (result object or None, commands run). Accepts stream-json (one object per line,
    the last `event: result` wins) and plain json (a single object with `response`)."""
    result: Optional[dict] = None
    commands: Dict[object, str] = {}
    for n, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if not isinstance(d, dict):
            continue
        if d.get("event") == "result" and isinstance(d.get("result"), dict):
            result = d["result"]
        elif "response" in d and "event" not in d:
            result = d
        elif d.get("event") == "step_update" and isinstance(d.get("step_update"), dict):
            su = d["step_update"]
            params = (su.get("tool_info") or {}).get("parameters") or {}
            cmd = params.get("CommandLine")
            if isinstance(cmd, str) and cmd.strip():
                commands[su.get("step_index", f"line{n}")] = cmd
    return result, list(commands.values())


def owner_approval(request_text: str) -> str:
    m = OWNER_LINE_RE.search(request_text)
    return m.group(1).strip().rstrip("*").strip() if m else ""


# ---------------------------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------------------------
def append_ledger(repo: Path, thread: str, rnd: str, req_sha: str, reply_sha: str, positions: str,
                  verdict: str, denied: int, tokens: str, copy_of: str) -> None:
    path = repo / LEDGER_REL
    if not path.exists():
        path.write_text(LEDGER_HEADER)
    row = "| " + " | ".join(_beta._md(c) for c in (
        thread, f"R{rnd}", _dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"),
        f"`{req_sha}`", f"`{reply_sha}`" if reply_sha != "—" else "—",
        positions or "—", verdict, str(denied), tokens or "—", copy_of)) + " |\n"
    with open(path, "a") as f:
        f.write(row)


# ---------------------------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------------------------
@dataclass
class Result:
    code: int
    verdict: str
    reply_rel: str = ""
    positions: Optional[List[_beta.Row]] = None
    violations: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    log_rel: str = ""
    run_dir: str = ""
    denied: int = 0
    tokens: str = ""


def _refuse(msg: str, reply_rel: str = "") -> Result:
    return Result(EXIT_REFUSED, VERDICTS[EXIT_REFUSED], reply_rel=reply_rel, notes=[msg])


def _main_checkout(repo: Path) -> Path:
    common = Path(_beta._git(repo, "rev-parse", "--path-format=absolute",
                             "--git-common-dir").decode().strip())
    return common.parent if common.name == ".git" else common


def run_relay(request: str, repo: Path, launcher: Launcher = agy_launcher,
              timeout_s: float = DEFAULT_TIMEOUT_MIN * 60,
              work_root: Optional[Path] = None, home: Optional[Path] = None) -> Result:
    repo = repo.resolve()
    home = (home or Path.home()).resolve()
    to_gemini = (repo / TO_GEMINI_REL).resolve()

    # 1. Validate the request.
    raw = Path(request)
    req = raw if raw.is_absolute() else Path.cwd() / raw
    if not req.exists() and not raw.is_absolute():
        req = repo / raw
    if req.is_symlink():
        return _refuse(f"request is a symlink: {request}")
    req = req.resolve()
    if req.parent != to_gemini:
        return _refuse(f"request is not directly inside {TO_GEMINI_REL}/: {request}")
    if not req.is_file():
        return _refuse(f"request does not exist: {request}")
    m = REQUEST_RE.match(req.name)
    if not m:
        return _refuse(f"request name does not match X-<id>_R<round>_ALPHA_<...>.md: {req.name}")
    thread, rnd = "X-" + m.group("id"), m.group("round")
    request_rel = req.relative_to(repo).as_posix()
    request_text = req.read_text(errors="replace")
    approval = owner_approval(request_text)
    if not approval:
        return _refuse("request has no `Owner approval:` line. Gemini reviews only items the owner "
                       "chose (owner ruling, 2026-11-26); state his approval for this use in the request")

    # 2. Expected reply; never overwritten.
    reply_rel = f"{FROM_GEMINI_REL}/{thread}_R{rnd}_GEMINI_REVIEW.md"
    reply = repo / reply_rel
    if reply.exists() or reply.is_symlink():
        return _refuse(f"reply already exists: {reply_rel}; a review is never overwritten", reply_rel)

    # 3. Where the copy goes. It must be outside the repository and every protected path.
    main_checkout = _main_checkout(repo)
    root = (work_root or Path(os.path.expanduser(os.environ.get(WORK_ROOT_ENV, WORK_ROOT_DEFAULT)))).resolve()
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    review_dir = root / f"{thread}_R{rnd}_{stamp}"
    copy_dir, run_dir = review_dir / "repo", review_dir / "relay"
    protected = protected_paths(repo, main_checkout, run_dir, home)
    for p in protected:
        if p != run_dir and (_is_under(copy_dir, p)):
            return _refuse(f"review copy folder {copy_dir} is inside a path Gemini may not read "
                           f"({p}); set {WORK_ROOT_ENV} to a folder outside it")
    if review_dir.exists():
        return _refuse(f"review folder already exists: {review_dir}")

    req_sha = _beta._sha_file(req)

    # 4. Build the copy and the lockdown.
    run_dir.mkdir(parents=True)
    info = build_copy(repo, copy_dir)
    settings_path = run_dir / "settings.json"
    settings_path.write_text(json.dumps(lockdown_settings(protected), indent=2) + "\n")
    before = digest_tree(copy_dir)
    copy_of = f"`{info.head[:12]}` + {info.dirty} uncommitted, {info.files} files"

    # 5. Run agy.
    log_dir = repo / LOG_DIR_REL
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{thread}_R{rnd}.log"
    prompt = build_prompt(request_rel)
    print_timeout_min = max(1, int(timeout_s // 60) - 2)
    out_path, err_path = run_dir / "out.jsonl", run_dir / "err.txt"
    notes: List[str] = []
    failed = False
    t0 = time.monotonic()
    with open(log_path, "a") as log_fh:
        log_fh.write(f"\n===== relay run {_dt.datetime.now().astimezone().isoformat()} "
                     f"request={request_rel} timeout_s={timeout_s}\n")
        log_fh.write("prompt: " + json.dumps(prompt) + "\n")
        log_fh.write("argv: " + json.dumps(agy_argv(prompt, settings_path, print_timeout_min)) + "\n")
        log_fh.write(f"review folder: {review_dir}\ncopy of: {copy_of}\n")
        log_fh.write("copy exclusions: " + json.dumps(list(COPY_EXCLUDED)) + "\n")
        log_fh.write("copy skipped: " + json.dumps(info.skipped) + "\n")
        log_fh.write("owner approval: " + json.dumps(approval) + "\n")
        log_fh.flush()
        with open(out_path, "wb") as out_fh, open(err_path, "wb") as err_fh:
            try:
                proc = launcher(prompt, copy_dir, settings_path, out_fh, err_fh, print_timeout_min)
            except Exception as e:  # failed to start
                failed = True
                notes.append(f"agy failed to start: {type(e).__name__}: {e}")
                proc = None
            if proc is not None:
                try:
                    rc = proc.wait(timeout=timeout_s)
                    if rc != 0:
                        failed = True
                        notes.append(f"agy exited with status {rc}")
                except subprocess.TimeoutExpired:
                    _beta._kill_group(proc)
                    failed = True
                    notes.append(f"agy timed out after {timeout_s:.0f}s; process group terminated "
                                 f"(final status {proc.returncode})")
        elapsed = time.monotonic() - t0

        # 6. Read the answer.
        result, commands = parse_agy_output(out_path.read_text(errors="replace"))
        (run_dir / "commands.log").write_text("".join(c.replace("\n", "\\n") + "\n" for c in commands))
        response = ""
        denied_actions: list = []
        tokens = ""
        conversation = ""
        if result is None:
            if not failed:
                failed = True
                notes.append("agy gave no result object")
        else:
            conversation = str(result.get("conversation_id", ""))
            status = str(result.get("status", ""))
            response = result.get("response") or ""
            denied_actions = result.get("denied_actions") or []
            total = (result.get("usage") or {}).get("total_tokens")
            tokens = str(total) if total is not None else ""
            if status != "SUCCESS":
                failed = True
                notes.append(f"agy status is {status!r}, not 'SUCCESS'")
            if not str(response).strip():
                failed = True
                notes.append("agy's answer is empty")
        if denied_actions:
            notes.append(f"DENIED ACTIONS: {len(denied_actions)}. Gemini was refused something it "
                         f"tried, so the review may be incomplete. Commands it ran: "
                         f"{run_dir / 'commands.log'}")

        # 7. The copy must be unchanged: Gemini is read-only.
        violations = compare_digests(before, digest_tree(copy_dir))

        # 8. File the reply (only a real answer is ever filed) and check its positions table.
        positions = None
        reply_sha = "—"
        malformed = ""
        if not failed:
            header = (
                f"# {thread} · R{rnd} · GEMINI REVIEW\n\n"
                f"Filed by `scripts/gemini_relay.py` from Gemini's answer, unedited below the line. "
                f"Not binding; holds nothing (owner ruling, 2026-11-26).\n\n"
                f"- Reviewer: Gemini `{GEMINI_MODEL}`, Antigravity CLI, mode `{AGY_MODE}`, read-only "
                f"on a copy\n"
                f"- Date: {_dt.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}\n"
                f"- Request: `{request_rel}` (sha256 `{req_sha}`)\n"
                f"- Copy reviewed: working tree at HEAD `{info.head}` with {info.dirty} uncommitted "
                f"or untracked entries, {info.files} files; left out: "
                f"{', '.join('`' + e + '`' for e in COPY_EXCLUDED)}\n"
                f"- Owner approval (from the request): {approval}\n"
                f"- Run: conversation `{conversation}`, {elapsed:.0f} s, tokens {tokens or 'unknown'}, "
                f"commands run {len(commands)}, denied actions {len(denied_actions)}\n"
                f"- Copy changed during the run: {'YES, see the relay output' if violations else 'no'}\n"
                f"\n---\n\n"
            )
            reply.parent.mkdir(parents=True, exist_ok=True)
            reply.write_text(header + str(response).rstrip() + "\n")
            reply_sha = _beta._sha_file(reply)
            positions, malformed = _beta.parse_positions(str(response))

        if violations:
            code = EXIT_VIOLATION
            notes.append("Gemini changed its review copy although it was told to change nothing. "
                         "The live tree is not checked by this relay. Nothing was reverted.")
        elif failed:
            code = EXIT_UNAVAILABLE
            notes.append(f"No reply was filed. Raw output: {out_path} and {err_path}")
        elif malformed:
            code = EXIT_MALFORMED
            notes.append(f"malformed: {malformed}. The answer is filed as Gemini wrote it.")
        else:
            code = EXIT_RECEIVED
        if malformed and code != EXIT_MALFORMED:
            notes.append(f"reply also malformed: {malformed}")

        log_fh.write(f"===== end elapsed_s={elapsed:.1f} verdict={VERDICTS[code]} "
                     f"notes={json.dumps(notes)}\n")

    # 9. Ledger.
    pos_str = "; ".join(f"{q} {p}" + (f" ({s})" if s else "") for q, p, s, _ in positions) if positions else ""
    append_ledger(repo, thread, rnd, req_sha, reply_sha, pos_str, VERDICTS[code],
                  len(denied_actions), tokens, copy_of)

    return Result(code, VERDICTS[code], reply_rel if reply_sha != "—" else "", positions, violations,
                  notes, elapsed, log_path.relative_to(repo).as_posix(), str(review_dir),
                  len(denied_actions), tokens)


def print_result(r: Result) -> None:
    print(f"VERDICT: {r.verdict} (exit {r.code})")
    if r.reply_rel:
        print(f"reply: {r.reply_rel}")
    if r.log_rel:
        print(f"log: {r.log_rel}  elapsed: {r.elapsed_s:.1f}s  tokens: {r.tokens or 'unknown'}")
    if r.run_dir:
        print(f"review folder: {r.run_dir}")
    if r.positions:
        print("| Question | Position | Severity |\n|---|---|---|")
        for q, p, s, why in r.positions:
            print(f"| {q} | {p} | {f'{s} — {why}' if s else '—'} |")
    for v in r.violations:
        print(f"CHANGED IN THE REVIEW COPY: {v}")
    for n in r.notes:
        print(f"note: {n}")
    if r.code == EXIT_RECEIVED:
        print("Gemini's positions are not binding and hold nothing. Read the reply file in full, "
              "verify each finding yourself, then report to Michael.")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Alpha→Gemini review relay (see docs/exchange/GEMINI_RELAY_PROTOCOL.md)")
    ap.add_argument("--request", required=True, help="docs/exchange/to_gemini/X-<id>_R<r>_ALPHA_<...>.md")
    ap.add_argument("--timeout-min", type=float, default=DEFAULT_TIMEOUT_MIN)
    a = ap.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    print("Left out of the review copy:", flush=True)
    for e in COPY_EXCLUDED:
        print(f"  - {e}", flush=True)
    r = run_relay(a.request, repo, timeout_s=a.timeout_min * 60)
    print_result(r)
    return r.code


if __name__ == "__main__":
    sys.exit(main())
