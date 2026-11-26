#!/usr/bin/env python3
"""tests/test_gemini_relay.py — gates for scripts/gemini_relay.py.

Every gate runs in its own temp git repository, created outside the TFM tree, with a fake agy
launcher. No gate calls the real agy and no gate spends Gemini quota. The fixture repository is
committed once by the test itself (owner ruling 2026-11-25 19:26: a test may commit inside a
temporary repository it creates outside the TFM tree).

Run:  python3 -u tests/test_gemini_relay.py
Prints one line per gate and `COMPLETION SENTINEL: PASS` only if every gate and every mutant ran
and passed.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIVE_REPO = HERE.parent
SCRIPTS = LIVE_REPO / "scripts"
RELAY_SRC = SCRIPTS / "gemini_relay.py"
BETA_SRC = SCRIPTS / "beta_relay.py"
TMP_ROOT: Path = None  # created by init_tmp() when the tests run, never at import

REQ_NAME = "X-0042_R1_ALPHA_REVIEW_REQUEST_test.md"
REQ_REL = f"docs/exchange/to_gemini/{REQ_NAME}"
REPLY_REL = "docs/exchange/from_gemini/X-0042_R1_GEMINI_REVIEW.md"
LEDGER_REL = "docs/exchange/GEMINI_LEDGER.md"
REQUEST_TEXT = "# X-0042 R1\n\n- Owner approval: Michael, 2026-11-26 08:00, \"yes, review it\"\n\n## Prior record\nnone\n\n1. Q1?\n2. Q2?\n"

GOOD = "| Question | Position | Severity |\n|---|---|---|\n| Q1 | NO OBJECTION | — |\n| Q2 | SUGGESTION | — |\n\n## Q1\nFine.\n"
NO_TABLE = "Looks fine to me.\n"
OBJ_NO_SEV = "| Question | Position | Severity |\n|---|---|---|\n| Q1 | OBJECTION | — |\n"
OBJ_LIVE = "| Question | Position | Severity |\n|---|---|---|\n| Q1 | OBJECTION | LIVE — wrong unit at a.py:1 |\n"


def init_tmp() -> Path:
    global TMP_ROOT
    if TMP_ROOT is None:
        TMP_ROOT = Path(tempfile.mkdtemp(prefix="gemini_relay_test_"))
    return TMP_ROOT


def load_module(src_text: str, name: str):
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    path = init_tmp() / f"{name}.py"
    path.write_text(src_text)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sh(cwd: Path, *args: str) -> str:
    return subprocess.run(list(args), cwd=str(cwd), capture_output=True, text=True, check=True).stdout


def stream(result: dict, commands=("grep -n foo a.py",)) -> str:
    lines = [json.dumps({"event": "init", "conversation_id": "c-1", "init": {"model": "m"}})]
    for i, c in enumerate(commands):
        lines.append(json.dumps({"event": "step_update", "step_update": {
            "conversation_id": "c-1", "step_index": i, "state": "ACTIVE", "step_type": "tool",
            "tool_name": "run_command",
            "tool_info": {"name": "run_command", "parameters": {"CommandLine": c}}}}))
    lines.append(json.dumps({"event": "result", "result": result}))
    return "\n".join(lines) + "\n"


def ok_result(response: str, **extra) -> dict:
    d = {"conversation_id": "c-1", "status": "SUCCESS", "response": response,
         "duration_seconds": 1.0, "num_turns": 2, "usage": {"total_tokens": 1234}}
    d.update(extra)
    return d


class Fx:
    """One temp repository, a fake home, a work root, and a recording fake launcher."""

    def __init__(self, mod, tag: str):
        self.mod = mod
        self.root = init_tmp() / tag
        self.repo = self.root / "repo"
        self.home = self.root / "home"
        self.work = self.root / "reviews"
        self.faketmp = self.root / "faketmp"
        for d in (self.repo / "scripts", self.repo / "docs/exchange/to_gemini",
                  self.repo / "docs/exchange/from_beta", self.home, self.faketmp):
            d.mkdir(parents=True)
        (self.repo / "a.py").write_text("x = 1\n")
        (self.repo / "AGENTS.md").write_text("You are Team Beta.\n")
        (self.repo / ".gitignore").write_text("logs/\nsecret.bin\n")
        (self.repo / "docs/exchange/from_beta/X-0042_R1_BETA_REVIEW.md").write_text("beta says\n")
        (self.repo / "docs/exchange/LEDGER.md").write_text("beta ledger\n")
        sh(self.repo, "git", "init", "-q")
        sh(self.repo, "git", "add", "-A")
        sh(self.repo, "git", "-c", "user.name=fixture", "-c", "user.email=fixture@localhost",
           "commit", "-q", "-m", "fixture")
        # after the commit: an uncommitted edit, an untracked file, an ignored file, the request
        (self.repo / "a.py").write_text("x = 2  # uncommitted\n")
        (self.repo / "new_untracked.py").write_text("y = 3\n")
        (self.repo / "secret.bin").write_text("ignored\n")
        (self.repo / REQ_REL).write_text(REQUEST_TEXT)
        mod.SYSTEM_PROTECTED = (str(self.faketmp),)
        self.calls = []

    def launcher(self, out_text: str = "", rc: int = 0, sleep_s: float = 0.0, touch=None):
        def launch(prompt, copy_dir, settings_path, out_fh, err_fh, print_timeout_min):
            self.calls.append({"prompt": prompt, "copy": Path(copy_dir), "settings": Path(settings_path),
                               "ptm": print_timeout_min})
            code = ("import sys,time,pathlib\n"
                    f"time.sleep({sleep_s})\n"
                    + (f"pathlib.Path({str(touch[0])!r}).write_text({touch[1]!r})\n" if touch else "")
                    + f"sys.stdout.write({out_text!r})\nsys.exit({rc})\n")
            return subprocess.Popen([sys.executable, "-c", code], cwd=str(copy_dir),
                                    stdin=subprocess.DEVNULL, stdout=out_fh, stderr=err_fh,
                                    start_new_session=True)
        return launch

    def run(self, launcher, request: str = REQ_REL, timeout_s: float = 60.0, work=None):
        return self.mod.run_relay(request, self.repo, launcher=launcher, timeout_s=timeout_s,
                                  work_root=work or self.work, home=self.home)

    def reply(self) -> Path:
        return self.repo / REPLY_REL

    def ledger(self) -> str:
        p = self.repo / LEDGER_REL
        return p.read_text() if p.exists() else ""


# --------------------------------------------------------------------------------------------- gates
def g_received(mod):
    fx = Fx(mod, "received")
    r = fx.run(fx.launcher(stream(ok_result(GOOD))))
    text = fx.reply().read_text() if fx.reply().exists() else ""
    rows = r.positions or []
    cmds = (Path(r.run_dir) / "relay" / "commands.log").read_text() if r.run_dir else ""
    ok = (r.code == 0 and r.verdict == "REVIEW RECEIVED" and r.reply_rel == REPLY_REL
          and GOOD.strip() in text and "GEMINI REVIEW" in text and "Not binding" in text
          and [(q, p) for q, p, _, _ in rows] == [("Q1", "NO OBJECTION"), ("Q2", "SUGGESTION")]
          and "| X-0042 | R1 |" in fx.ledger() and "REVIEW RECEIVED" in fx.ledger()
          and "1234" in fx.ledger() and "grep -n foo a.py" in cmds and len(fx.calls) == 1)
    return ok, f"code={r.code} rows={len(rows)} reply={fx.reply().exists()} cmds={cmds.strip()!r}"


def g_wrong_folder(mod):
    fx = Fx(mod, "wrongfolder")
    (fx.repo / "docs/exchange/to_beta").mkdir()
    other = fx.repo / "docs/exchange/to_beta" / REQ_NAME
    other.write_text(REQUEST_TEXT)
    r = fx.run(fx.launcher(stream(ok_result(GOOD))), request=f"docs/exchange/to_beta/{REQ_NAME}")
    ok = r.code == 4 and not fx.calls and not fx.reply().exists() and fx.ledger() == ""
    return ok, f"code={r.code} calls={len(fx.calls)}"


def g_no_owner(mod):
    fx = Fx(mod, "noowner")
    (fx.repo / REQ_REL).write_text("# X-0042 R1\n\n1. Q1?\n")
    r = fx.run(fx.launcher(stream(ok_result(GOOD))))
    ok = (r.code == 4 and not fx.calls and not fx.reply().exists() and fx.ledger() == ""
          and any("Owner approval" in n for n in r.notes))
    return ok, f"code={r.code} calls={len(fx.calls)}"


def g_reply_exists(mod):
    fx = Fx(mod, "replyexists")
    fx.reply().parent.mkdir(parents=True)
    fx.reply().write_text("earlier review\n")
    before = sha(fx.reply())
    r = fx.run(fx.launcher(stream(ok_result(GOOD))))
    ok = r.code == 4 and not fx.calls and sha(fx.reply()) == before
    return ok, f"code={r.code} calls={len(fx.calls)} unchanged={sha(fx.reply()) == before}"


def g_bad_name(mod):
    fx = Fx(mod, "badname")
    bad = fx.repo / "docs/exchange/to_gemini/review_me.md"
    bad.write_text(REQUEST_TEXT)
    r = fx.run(fx.launcher(stream(ok_result(GOOD))), request="docs/exchange/to_gemini/review_me.md")
    ok = r.code == 4 and not fx.calls
    return ok, f"code={r.code}"


def g_nonzero_exit(mod):
    fx = Fx(mod, "nonzero")
    r = fx.run(fx.launcher(stream(ok_result(GOOD)), rc=7))
    ok = (r.code == 2 and not fx.reply().exists() and "UNAVAILABLE" in fx.ledger()
          and any("status 7" in n for n in r.notes))
    return ok, f"code={r.code} reply={fx.reply().exists()}"


def g_no_result(mod):
    fx = Fx(mod, "noresult")
    r = fx.run(fx.launcher("not json at all\n"))
    ok = r.code == 2 and not fx.reply().exists() and "UNAVAILABLE" in fx.ledger()
    return ok, f"code={r.code}"


def g_bad_status(mod):
    fx = Fx(mod, "badstatus")
    r = fx.run(fx.launcher(stream(ok_result(GOOD, status="ERROR"))))
    ok = r.code == 2 and not fx.reply().exists()
    return ok, f"code={r.code} reply={fx.reply().exists()}"


def g_empty_answer(mod):
    fx = Fx(mod, "empty")
    r = fx.run(fx.launcher(stream(ok_result("   \n"))))
    ok = r.code == 2 and not fx.reply().exists()
    return ok, f"code={r.code} reply={fx.reply().exists()}"


def g_timeout(mod):
    fx = Fx(mod, "timeout")
    r = fx.run(fx.launcher(stream(ok_result(GOOD)), sleep_s=30), timeout_s=1.0)
    ok = (r.code == 2 and not fx.reply().exists() and r.elapsed_s < 20
          and any("timed out" in n for n in r.notes))
    return ok, f"code={r.code} elapsed={r.elapsed_s:.1f}"


def g_no_table(mod):
    fx = Fx(mod, "notable")
    r = fx.run(fx.launcher(stream(ok_result(NO_TABLE))))
    ok = r.code == 5 and fx.reply().exists() and NO_TABLE.strip() in fx.reply().read_text()
    return ok, f"code={r.code}"


def g_objection_needs_severity(mod):
    fx = Fx(mod, "nosev")
    r = fx.run(fx.launcher(stream(ok_result(OBJ_NO_SEV))))
    fx2 = Fx(mod, "livesev")
    r2 = fx2.run(fx2.launcher(stream(ok_result(OBJ_LIVE))))
    ok = (r.code == 5 and r2.code == 0 and r2.positions[0][1:3] == ("OBJECTION", "LIVE")
          and "(LIVE)" in fx2.ledger())
    return ok, f"no-severity code={r.code}, LIVE code={r2.code}"


def g_copy_modified(mod):
    fx = Fx(mod, "modified")
    r = fx.run(fx.launcher(stream(ok_result(GOOD)), touch=("a.py", "x = 99\n")))
    fx2 = Fx(mod, "created")
    r2 = fx2.run(fx2.launcher(stream(ok_result(GOOD)), touch=("sneaky.txt", "hi\n")))
    ok = (r.code == 3 and any("modified: a.py" in v for v in r.violations)
          and r2.code == 3 and any("created: sneaky.txt" in v for v in r2.violations)
          and (fx.repo / "a.py").read_text() == "x = 2  # uncommitted\n"
          and not (fx2.repo / "sneaky.txt").exists())
    return ok, f"modified code={r.code} {r.violations}, created code={r2.code} {r2.violations}"


def g_copy_contents(mod):
    fx = Fx(mod, "contents")
    fx.run(fx.launcher(stream(ok_result(GOOD))))
    c = fx.calls[0]["copy"]
    have = sorted(p.relative_to(c).as_posix() for p in c.rglob("*") if p.is_file())
    ok = ((c / "a.py").read_text() == "x = 2  # uncommitted\n"          # uncommitted edit included
          and (c / "new_untracked.py").exists()                          # untracked included
          and (c / REQ_REL).read_text() == REQUEST_TEXT                  # the request itself
          and not (c / "secret.bin").exists()                            # git-ignored left out
          and not (c / "docs/exchange/from_beta").exists()               # Beta's replies left out
          and not (c / "docs/exchange/LEDGER.md").exists()
          and not (c / ".git").exists()
          and len(have) == 5)
    return ok, f"{len(have)} files in copy: {have}"


def g_lockdown(mod):
    fx = Fx(mod, "lockdown")
    fx.run(fx.launcher(stream(ok_result(GOOD))))
    call = fx.calls[0]
    d = json.loads(call["settings"].read_text())
    deny, allow = d["permissions"]["deny"], d["permissions"]["allow"]
    reads = [x[len("read_file("):-1] for x in deny if x.startswith("read_file(")]
    need = [str(fx.repo.resolve()), str(fx.home.resolve() / ".config"), str(fx.home.resolve() / ".ssh"),
            str(fx.home.resolve() / "agy_trial/keys"), str(fx.faketmp), str(call["settings"].parent)]
    ok = (d.get("enableTerminalSandbox") is True and allow == ["command(*)"]
          and all(Path(x).is_absolute() for x in reads) and len(reads) == 3 + len(mod.HOME_PROTECTED)
          and all(n in reads for n in need)
          and all(c in deny for c in mod.COMMAND_DENY) and len(mod.COMMAND_DENY) == 11
          and not any(str(call["copy"]).startswith(r + "/") or str(call["copy"]) == r
                      for r in reads if r != str(call["settings"].parent)))
    return ok, f"{len(reads)} read rules, all absolute={all(Path(x).is_absolute() for x in reads)}, {len(deny) - len(reads)} command rules"


def g_prompt_and_argv(mod):
    fx = Fx(mod, "prompt")
    fx.run(fx.launcher(stream(ok_result(GOOD))), timeout_s=600.0)
    call = fx.calls[0]
    argv = mod.agy_argv("P", Path("/s.json"), 8)
    p = call["prompt"]
    ok = (p == mod.build_prompt(REQ_REL) and REQ_REL in p
          and "Do not edit, create or delete any file" in p and "Never commit or push" in p
          and "Not handled" in p and "compares nothing" in p and "At most 10 findings" in p
          and "smallest change" in p and "no links and no absolute paths" in p and call["ptm"] == 8
          and argv == ["agy", "-p", "P", "--settings", "/s.json", "--mode", "plan", "--model",
                       "gemini-4-hydrogen-xhigh", "--output-format", "stream-json",
                       "--print-timeout", "8m"])
    return ok, f"prompt {len(p)} chars, print-timeout {call['ptm']}m"


def g_denied_actions(mod):
    fx = Fx(mod, "denied")
    r = fx.run(fx.launcher(stream(ok_result(GOOD, denied_actions=[{"action": "command", "display_name": "Bash"}]))))
    row = [ln for ln in fx.ledger().splitlines() if ln.startswith("| X-0042")]
    ok = (r.code == 0 and r.denied == 1 and any("DENIED ACTIONS: 1" in n for n in r.notes)
          and len(row) == 1 and "| REVIEW RECEIVED | 1 |" in row[0]
          and "denied actions 1" in fx.reply().read_text())
    return ok, f"code={r.code} denied={r.denied}"


def g_work_root_protected(mod):
    fx = Fx(mod, "rootinside")
    r = fx.run(fx.launcher(stream(ok_result(GOOD))), work=fx.repo / "reviews_here")
    fx2 = Fx(mod, "rootintmp")
    r2 = fx2.run(fx2.launcher(stream(ok_result(GOOD))), work=fx2.faketmp / "reviews")
    ok = r.code == 4 and not fx.calls and r2.code == 4 and not fx2.calls
    return ok, f"inside repo code={r.code}, inside protected tmp code={r2.code}"


def g_live_tree(mod):
    fx = Fx(mod, "livetree")
    before = sh(fx.repo, "git", "status", "--porcelain=v1", "--untracked-files=all").splitlines()
    head = sh(fx.repo, "git", "rev-parse", "HEAD")
    fx.run(fx.launcher(stream(ok_result(GOOD))))
    after = sh(fx.repo, "git", "status", "--porcelain=v1", "--untracked-files=all").splitlines()
    new = sorted(set(after) - set(before))
    ok = (new == [f"?? {LEDGER_REL}", f"?? {REPLY_REL}"] and set(before) <= set(after)
          and sh(fx.repo, "git", "rev-parse", "HEAD") == head
          and (fx.repo / "logs/gemini_relay/X-0042_R1.log").is_file())
    return ok, f"new entries in the tree: {new}"


def g_plain_json(mod):
    fx = Fx(mod, "plainjson")
    r = fx.run(fx.launcher(json.dumps(ok_result(GOOD)) + "\n"))
    ok = r.code == 0 and fx.reply().exists()
    return ok, f"code={r.code}"


def g_beta_untouched(mod):
    import beta_relay
    ok = (beta_relay.TO_BETA_REL == "docs/exchange/to_beta" and beta_relay.FROM_BETA_REL == "docs/exchange/from_beta"
          and mod.TO_GEMINI_REL != beta_relay.TO_BETA_REL and mod.FROM_GEMINI_REL != beta_relay.FROM_BETA_REL
          and mod.LEDGER_REL != beta_relay.LEDGER_REL and mod.LOG_DIR_REL != beta_relay.LOG_DIR_REL
          and mod.EXIT_REFUSED == beta_relay.EXIT_REFUSED == 4)
    return ok, "separate folders, ledger and log; same exit codes"


GATES = [
    ("GR-A", "a good answer is filed, parsed and ledgered", g_received),
    ("GR-B", "a request outside to_gemini/ is refused, agy not run", g_wrong_folder),
    ("GR-C", "a request without the owner's approval is refused", g_no_owner),
    ("GR-D", "an existing reply is never overwritten", g_reply_exists),
    ("GR-E", "a badly named request is refused", g_bad_name),
    ("GR-F", "agy exit status non-zero is UNAVAILABLE, nothing filed", g_nonzero_exit),
    ("GR-G", "no result object is UNAVAILABLE", g_no_result),
    ("GR-H", "status other than SUCCESS is UNAVAILABLE", g_bad_status),
    ("GR-I", "an empty answer is UNAVAILABLE", g_empty_answer),
    ("GR-J", "a timeout kills agy and is UNAVAILABLE", g_timeout),
    ("GR-K", "an answer with no positions table is MALFORMED", g_no_table),
    ("GR-L", "an OBJECTION needs LIVE or THEORETICAL", g_objection_needs_severity),
    ("GR-M", "a changed review copy is a BOUNDARY VIOLATION; live tree untouched", g_copy_modified),
    ("GR-N", "the copy holds the working tree and leaves out Beta's replies", g_copy_contents),
    ("GR-O", "lockdown: absolute read rules, the trial's command rules", g_lockdown),
    ("GR-P", "the fixed prompt and agy arguments", g_prompt_and_argv),
    ("GR-Q", "denied actions are counted and flagged", g_denied_actions),
    ("GR-R", "a copy folder inside a protected path is refused", g_work_root_protected),
    ("GR-S", "the live tree gains only the reply, the ledger and the log", g_live_tree),
    ("GR-T", "plain json output is accepted too", g_plain_json),
    ("GR-U", "separate from the Beta relay", g_beta_untouched),
]

# (name, description, old text, new text, gates that must turn red)
MUTANTS = [
    ("MUT-1", "owner-approval check removed", "    if not approval:\n", "    if False:\n", ["GR-C"]),
    ("MUT-2", "folder check removed", "    if req.parent != to_gemini:\n", "    if False:\n", ["GR-B"]),
    ("MUT-3", "reply-exists check removed", "    if reply.exists() or reply.is_symlink():\n", "    if False:\n", ["GR-D"]),
    ("MUT-4", "copy comparison disabled", "    violations = compare_digests(before, digest_tree(copy_dir))\n",
     "    violations = []\n", ["GR-M"]),
    ("MUT-5", "copy exclusions disabled", "    return any(rel == e or (e.endswith(\"/\") and rel.startswith(e)) for e in COPY_EXCLUDED)\n",
     "    return False\n", ["GR-N"]),
    ("MUT-6", "status check removed", "            if status != \"SUCCESS\":\n", "            if False:\n", ["GR-H"]),
    ("MUT-7", "empty-answer check removed", "            if not str(response).strip():\n", "            if False:\n", ["GR-I"]),
    ("MUT-8", "exit status ignored", "                    if rc != 0:\n", "                    if False:\n", ["GR-F"]),
    ("MUT-9", "protected-folder check removed", "        if p != run_dir and (_is_under(copy_dir, p)):\n",
     "        if False:\n", ["GR-R"]),
    ("MUT-10", "denied actions not flagged", "        if denied_actions:\n", "        if False:\n", ["GR-Q"]),
    ("MUT-11", "git-ignored files copied", "\"ls-files\", \"-z\", \"--cached\", \"--others\", \"--exclude-standard\"",
     "\"ls-files\", \"-z\", \"--cached\", \"--others\"", ["GR-N"]),
]


def run_gate(mod, name):
    fn = {n: f for n, _, f in GATES}[name]
    try:
        return fn(mod)
    except Exception as e:
        return False, f"raised {type(e).__name__}: {e}"


def main() -> int:
    init_tmp()
    src = RELAY_SRC.read_text()
    beta_before = sha(BETA_SRC)
    mod = load_module(src, "gemini_relay_under_test")
    results = []
    for name, desc, _ in GATES:
        ok, detail = run_gate(mod, name)
        results.append((name, ok))
        print(f"  [{'PASS' if ok else 'FAIL':<5}] {name:<5} — {desc}: {detail}", flush=True)

    for i, (mname, desc, old, new, targets) in enumerate(MUTANTS):
        applied = src.count(old) == 1
        reds = {}
        if applied:
            mmod = load_module(src.replace(old, new), f"gemini_relay_mutant_{i}")
            reds = {t: not run_gate(mmod, t)[0] for t in targets}
        ok = applied and all(reds.values())
        results.append((mname, ok))
        print(f"  [{'PASS' if ok else 'FAIL':<5}] {mname:<6} — {desc}: "
              + (", ".join(f"{t} {'red' if v else 'GREEN (not detected)'}" for t, v in reds.items())
                 if applied else f"MUTANT NOT APPLIED (text found {src.count(old)} times)"), flush=True)

    ok = sha(BETA_SRC) == beta_before
    results.append(("HYG-BETA", ok))
    print(f"  [{'PASS' if ok else 'FAIL':<5}] HYG-BETA — scripts/beta_relay.py unchanged by the suite", flush=True)

    shutil.rmtree(TMP_ROOT, ignore_errors=True)
    passed = sum(ok for _, ok in results)
    print(f"  {passed}/{len(results)} PASS", flush=True)
    if passed == len(results) == len(GATES) + len(MUTANTS) + 1:
        print("COMPLETION SENTINEL: PASS", flush=True)
        return 0
    print("COMPLETION SENTINEL: FAIL", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
