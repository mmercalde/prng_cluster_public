# Alpha–Gemini review relay — protocol

Status: in force once landed on main with Michael's yes. Tool: `scripts/gemini_relay.py`.
Owner ruling of 2026-11-26 06:34–06:35 governs: Gemini is an occasional second reviewer on items
Michael chooses, next to Beta; read-only and locked down; tightly scoped prompts; it writes no code
that lands; its reviews are not binding. Verbatim ruling on VM101:
`logs/e2e_session/OWNER_RULING_gemini_role_20261126.txt`. Trial record and CLI quirks:
project doc `claude/GEMINI_ANTIGRAVITY_USAGE_20261126.md`.

Beta stays the standing reviewer. `docs/exchange/RELAY_PROTOCOL.md` and `scripts/beta_relay.py`
are unchanged by this protocol.

## 1. Roles

- **Alpha** (Claude) writes the request.
- **A session** (Claude in chat, or Claude Code) runs the relay. It is Alpha's hands.
- **Gemini** (Google Antigravity CLI `agy`, model `gemini-4-hydrogen-xhigh`) reviews a copy.
- **Michael** chooses each item Gemini reviews, and decides what is done with every finding.

## 2. A review

1. **Michael says yes to this use.** Gemini is never run on a session's own initiative.
2. The session writes the request in `docs/exchange/to_gemini/`, named like a Beta request,
   `X-<id>_R<round>_ALPHA_<TYPE>_<topic>.md`, in the same layout (`AGENTS.md` §5): `## Prior record`,
   base commit, files under review, numbered questions. It adds one line the relay requires:

   ```
   - Owner approval: Michael, <date and time>, "<his words>"
   ```

   The questions name the exact gap and say what is out of scope. Gemini widens loose goals.
3. It runs the relay and waits:

   ```
   python3 scripts/gemini_relay.py --request docs/exchange/to_gemini/X-<id>_R<round>_ALPHA_<TYPE>_<topic>.md
   ```

4. It reads the reply file in `docs/exchange/from_gemini/` **in full**, then verifies each finding
   itself (read the code at the cited line; reproduce where it can) before reporting it.
5. Its report to Michael states, per question: Gemini's position, what the session verified, and
   what it could not verify. A finding the session confirmed is registered the way G-1 and G-2
   were; a finding it could not confirm is listed as unchecked.

**No Gemini position holds a commit.** `OBJECTION · LIVE` from Gemini means "verify this first",
not "stop". Nothing from Gemini is written to `docs/exchange/DEFERRED.md`. To send the same item
to Beta as well, write a separate request in `to_beta/` and run `beta_relay.py`; the two relays do
not share folders, ledgers or logs.

## 3. What the relay does and what its verdicts mean

The relay validates the request and refuses if the reply already exists. It then:

- **Builds a review copy** of the working tree it is run from, as it is at that moment: tracked
  files with their current content plus untracked files git does not ignore. It makes no commit.
  Left out, so the review is independent: `docs/exchange/from_beta/`, `docs/exchange/from_gemini/`,
  `docs/exchange/LEDGER.md`, `docs/exchange/DEFERRED.md`, `docs/exchange/GEMINI_LEDGER.md`.
  Symlinks are not copied. The copy goes to `~/gemini_reviews/<thread>_R<round>_<time>/repo`
  (override: `GEMINI_RELAY_WORKDIR`), which must be outside the repository and every protected path.
- **Writes the lockdown** for that run only, passed with `--settings`; Michael's own
  `settings.json` is never touched. Gemini may not read the TFM tree, the worktrees, the inbox,
  `~/.config` (API key), `~/.ssh`, `~/.claude`, `~/.codex`, the api-agent state, `/tmp`, the trial's
  key and setup folders, or the relay's own run folder. Commit, push, `sudo`, `ssh`, `scp`, `rsync`,
  `rm -rf`, `curl`, `wget`, `--run-pipeline`, and running tests or pipeline programs are denied.
  The command rules are the trial's, word for word.
- **Runs `agy`** on the copy with a fixed prompt built only in the relay. The prompt carries the
  owner's five prompt rules: answer only the numbered questions; smallest change, at most ten
  findings; check that every test or gate really compares something; list what was not handled;
  never edit, commit, push or launch anything.
- **Checks the copy is unchanged**, checks the positions table (same words and severity rule as
  Beta), **writes the reply file** from Gemini's answer under a short header, and appends one row to
  `docs/exchange/GEMINI_LEDGER.md`.

| Exit | Verdict | Meaning | What the session does |
|---|---|---|---|
| 0 | `REVIEW RECEIVED` | Reply filed and well-formed; the copy was not changed | Follows §2.4 |
| 2 | `UNAVAILABLE` | `agy` failed to start, exited non-zero, timed out, or gave no answer. Nothing is filed | Reports to Michael. **Never treated as agreement.** May run again |
| 3 | `BOUNDARY VIOLATION` | Gemini changed its copy, though told to change nothing | Stops and reports to Michael |
| 4 | `REFUSED` | Bad request path or name, no `Owner approval:` line, reply already exists, or the copy folder is inside a protected path. `agy` was not run | Fixes its request, or reports to Michael |
| 5 | `MALFORMED REVIEW` | Answer filed, but it has no valid positions table | Reads the file anyway and reports to Michael; the file stays as Gemini wrote it |

**Denied actions.** If Gemini was refused something it tried, the relay prints
`DENIED ACTIONS: <n>` and records the count in the reply header and the ledger. The review may be
incomplete. Every shell command Gemini ran is in `commands.log` in the run folder.

## 4. What stays with Michael

- The choice of every item Gemini reviews.
- What is done with each finding.
- Every commit and push, as always.
- Every `BOUNDARY VIOLATION`.

## 5. What a session must never do

- Run Gemini for a review by any route other than the relay, or add words to the relay's prompt.
- Run the relay without Michael's yes for that item, or write an `Owner approval:` line he did not give.
- Edit, move or delete a file in `docs/exchange/from_gemini/`.
- Report a Gemini finding as fact before verifying it, or summarize the review in place of the file.
- Treat `UNAVAILABLE`, a timeout or a malformed review as agreement.
- Let Gemini write code that lands.
- Read the Antigravity sign-in token, or turn on AI credits.

## 6. The ledger

`docs/exchange/GEMINI_LEDGER.md` is written only by the relay: one row per run that launched `agy`
(thread, round, date, request sha256, reply sha256, positions, verdict, denied actions, tokens, and
what the copy was made from). A `REFUSED` run writes no row.

## 7. Not covered

- **The live tree is not snapshotted.** Unlike the Beta relay, this one does not compare the TFM
  tree before and after, so other sessions may work while it runs. Gemini is kept out of the live
  tree by the lockdown and the terminal sandbox, and those match commands and paths as written:
  they stop the ordinary case and are not a cage.
- **Quota.** Gemini runs on Michael's flat Google plan: quota, not money. The relay records tokens
  per review. It cannot read the remaining quota; check with `/usage` inside `agy`. AI credits stay off.
- **Backgrounded commands.** If Gemini backgrounds a command, `agy` ends the session. The prompt
  tells it to work in the foreground. A run cut short this way is `UNAVAILABLE` or incomplete.
- **A dirty tree.** The copy is whatever the working tree holds when the relay starts, including
  another session's half-finished edits. Run it from the worktree that holds the work under review.
- Gemini cannot run tests. A finding that needs a run to confirm is the session's to reproduce.
