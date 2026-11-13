# Alpha–Beta review relay — protocol

Status: in force from 2026-11-12. Tool: `scripts/beta_relay.py`. Beta's contract: `AGENTS.md`.
Owner rulings of 2026-11-11 govern authority: Beta reviews, it does not approve, and its reviews
are not binding. Michael decides.

## 1. Roles

- **Alpha** (Claude, in chat) writes briefs.
- **A Claude Code session** carries a brief out. It is Alpha's hands on this machine.
- **Beta** (Codex, under `AGENTS.md`) reviews.
- **Michael** commits, pushes, and decides every disagreement.

## 2. A review cycle

1. The Claude Code session finishes its work and stops editing.
2. It writes the request file in `docs/exchange/to_beta/`, named
   `X-<id>_R<round>_ALPHA_<TYPE>_<topic>.md`, in the layout `AGENTS.md` §5 expects: review
   class, base commit, files under review, and numbered questions with the allowed answers
   (`NO OBJECTION` · `OBJECTION` · `SUGGESTION` · `NEEDS EVIDENCE`). When the brief supplies the
   request file, the session uses it unchanged.
3. It runs the relay and waits:

   ```
   python3 scripts/beta_relay.py --request docs/exchange/to_beta/X-<id>_R<round>_ALPHA_<TYPE>_<topic>.md
   ```

   **No other session writes to the working tree while the relay runs.** The relay compares the
   tree before and after; it cannot tell Beta's writes from another session's.
4. It reads Beta's review file in full and acts by this table:

   | Beta's position | What the session does |
   |---|---|
   | `NO OBJECTION` | Records it and continues. |
   | `NEEDS EVIDENCE` | Supplies the evidence in `X-<id>_R<n+1>_ALPHA_REPLY.md` and runs the relay again. |
   | `SUGGESTION` | Adopts it only if it is inside the brief's stated scope and allowed files; otherwise lists it for Alpha. Says which in its reply file. |
   | `OBJECTION` · `LIVE` | If the session agrees the work is wrong **and** the fix is inside the brief's scope, it fixes, re-tests, and asks again. **If it disagrees, or the fix is outside scope, it stops.** It never overrules Beta and never argues Beta into agreement. A `LIVE` objection holds the commit. |
   | `OBJECTION` · `THEORETICAL` | Records it and continues. The relay has already appended it to `docs/exchange/DEFERRED.md`. The session lists it in its final report. It holds nothing. |

   Every `OBJECTION` carries a severity (owner ruling, 2026-11-12; `AGENTS.md` §5):

   - **`LIVE`** means it would cause a wrong result in a realistic run on this system as it is
     today.
   - **`THEORETICAL`** means it needs conditions that do not exist here today, such as a
     submodule, a hostile process, or a configuration this repository does not use.

   An `OBJECTION` row without a severity is a `MALFORMED REVIEW` (exit 5).

   **Alpha or the session may contest a label.** The session says so in its reply file and in
   its final report, with both positions. A contested label goes to Michael. Until he rules,
   the session treats the objection as Beta labelled it.

5. **At most three rounds per thread.** After the third, the session stops.
6. The session's final report to Michael states, per question: Beta's last position, what was
   done about it, and anything unresolved with both positions side by side.

## 3. What the relay does and what its verdicts mean

The relay validates the request and refuses if the reply already exists. It then takes a
**snapshot**, runs Codex with the fixed prompt `Read <request path> and reply as it asks.`
(reasoning effort `high`), takes the snapshot again, checks the reply's positions table, and
appends one row to `docs/exchange/LEDGER.md`.

**What the snapshot covers.** Everything here is outside `docs/exchange/from_beta/`:

- `HEAD`, and a digest of every git ref (`git for-each-ref`).
- `git status --porcelain --untracked-files=all` and the sha256 of `git diff HEAD --binary`.
- The sha256 of every tracked file that `git diff HEAD --name-only` reports as differing. A
  file flagged `assume-unchanged` or `skip-worktree` is not reported, so it is not hashed.
- The sha256 of every untracked, non-ignored file that `git ls-files --others` lists. A
  symlink is hashed through to its target's content; the link's own target path is not
  recorded. A nested git repository is listed as a single directory entry, and its contents
  are not read. The same holds for an ignored nested repository below.
- **Every git-ignored file in the repository**, except the exclusions listed below. Files of
  8 MiB or less are recorded by sha256; larger files by size and mtime in nanoseconds. A
  symlink is recorded by its target.
- Whether the live halt file exists (existence only, not content). Its path is read from
  `agents/safety/kill_switch.py` (`KillSwitch.halt_file_path`).
- The live miner ledger in the `staging_dir` named by `agent_manifests/window_optimizer.json`:
  the plain-byte sha256 of `miner_ledger.db`, and the size and sha256 of any `-wal`, `-shm` or
  `-journal` file beside it. The ledger is never opened with SQLite.

**Excluded ignored paths** (printed on every run, and written to the run log):

- any `__pycache__/` directory
- any `*.pyc` file
- any `.pytest_cache/` directory
- `logs/beta_relay/`, the relay's own logs

In `from_beta/`, every file is hashed. Changes other than the expected reply are listed and
recorded in the ledger, and are not violations.

| Exit | Verdict | Meaning | What the session does |
|---|---|---|---|
| 0 | `REVIEW RECEIVED` | Reply present and well-formed, and **no change was detected by the snapshot above**. It does not cover what §7 lists | Reads the file and follows §2.4 |
| 2 | `UNAVAILABLE` | Codex failed to start or sign in, exited non-zero, timed out, or wrote no reply | Stops and reports to Michael. **Never treated as agreement** |
| 3 | `BOUNDARY VIOLATION` | The snapshot detected a change outside `from_beta/` during the run. The relay lists it and reverts nothing | Stops and reports to Michael. Touches nothing it lists |
| 4 | `REFUSED` | Bad request path or the reply already exists; Codex was not run | Fixes its own request path, or reports to Michael if the reply exists |
| 5 | `MALFORMED REVIEW` | Reply exists but has no positions table, uses another word, or has an `OBJECTION` row without a `LIVE`/`THEORETICAL` severity and a reason | Stops and reports to Michael. The file stays as Beta wrote it |

If a run ends `UNAVAILABLE` or `BOUNDARY VIOLATION` but Beta did write the reply file, the next
run of the same round is `REFUSED` (a review is never overwritten). What happens to that thread
is Michael's decision.

Beta may also write other files in `from_beta/` (for example `X-<id>_R1_BETA_FINDING.md` to
open its own thread). The relay lists them and records them in the ledger; it is not a violation.

## 4. What stays with Michael

- Every commit and push. For BEFORE-COMMIT work the session stops after Beta's review, and
  Michael commits only then.
- Every unresolved `LIVE` `OBJECTION`, which holds the commit. `THEORETICAL` objections hold
  nothing; they are in `docs/exchange/DEFERRED.md` and the session's final report.
- Every contested severity label.
- Every `BOUNDARY VIOLATION`, `UNAVAILABLE` or `MALFORMED REVIEW` verdict.

## 5. What a session must never do

- Edit, move or delete a file in `docs/exchange/from_beta/`.
- Add words to the relay's prompt, or run Codex for a review by any route other than the relay.
- Summarize or restate Beta's review in place of the file. Reports point at the file.
- Treat `UNAVAILABLE`, a timeout or a malformed review as agreement.
- Run the relay while another session is writing to the working tree.

## 6. The ledger

`docs/exchange/LEDGER.md` is written only by the relay: one row per relay run that launched
Codex (thread, round, date, request sha256, reply sha256, positions per question with the
severity of each `OBJECTION`, verdict, other `from_beta/` changes). A `REFUSED` run launches
nothing and writes no row. The ledger opens with one hand-entered row recording thread X-0000 as
closed on 2026-11-12, with Beta's final position on Q4 `NO OBJECTION` given in session, not in a
file.

`docs/exchange/DEFERRED.md` is also written only by the relay. It gets one row per `THEORETICAL`
objection in a well-formed review (thread, round, question, date, and Beta's one-sentence
reason), whatever the run's verdict. Rows written before 2026-11-12's severity ruling are not
back-filled.

## 7. Not covered

The relay detects changes after the fact. It never prevents or reverts one. The Codex sandbox
it uses (`workspace-write`, writable roots: the repository, `/tmp`, `$TMPDIR`) allows more than
`AGENTS.md` does. **The snapshot does not see:**

1. Changes under the four excluded ignored paths in §3.
2. An ignored file larger than 8 MiB whose content changes while its size and mtime stay the
   same. Resetting an mtime after a same-size write is possible.
3. Mode or permission changes to untracked or ignored files, since only their content is
   hashed. **Permission changes to tracked files are not checked either.** Git records only the
   executable bit, and only when `core.filemode=true`, which is this repository's setting on
   2026-11-12. A read/write change such as `0644→0600` goes unseen. With `core.filemode=false`,
   an executable-bit change goes unseen too. (Found by Beta, X-RELAY-CHECK R3.)
4. Empty directories created or removed. Git lists files only.
5. Anything inside `.git/` other than `HEAD` and refs, for example `.git/config`, `.git/hooks/`
   or the index. Index changes show up only as far as they change `git status`. Whether the
   sandbox can write `.git/` at all has not been verified.
6. Anything outside the repository other than the live halt file's existence and the four
   ledger files. That includes the rest of `~/miner_staging/`, other files in `/tmp` (which
   `AGENTS.md` §3 permits for throwaway checks), and the rest of the home directory.
7. A change made and undone within the run, between the two snapshots.
8. Effects that leave no file behind: processes started, or network use.
9. **Who made a change.** A concurrent session's write is indistinguishable from Beta's. Hence
   the rule that no other session writes during a run.
10. **Retargeting an untracked, non-ignored symlink** to a file with identical content. Only the
    content reached through the link is hashed. Ignored symlinks are recorded by target and are
    covered.
11. **Editing a tracked file that was flagged `assume-unchanged` or `skip-worktree`** before the
    run. Git does not report it, so it is never hashed.
12. **Editing a file inside a nested git repository, untracked or ignored.** Git lists its
    directory as one entry, and the files inside are not read. Beta verified the untracked case;
    Alpha's session checked that an ignored nested repository is listed the same way.
13. **Changing the contents of a halt file that already existed.** Only existence is checked.
14. **Further edits inside a tracked submodule that already has uncommitted changes.** Submodule
    contents are not hashed. Once the outer `git status` shows the submodule as modified, more
    edits inside it do not change what the snapshot records. The repository had no submodules on
    2026-11-12. (Found by Beta, X-RELAY-CHECK R3.)

Items 10–13 were found and verified by Beta in X-RELAY-CHECK R2. None of their preconditions
(an assume-unchanged or skip-worktree flag, an untracked symlink, an untracked or ignored nested
repository) existed in the live tree on 2026-11-12. Whether to detect them, rather than disclose
them, is open for Alpha.
