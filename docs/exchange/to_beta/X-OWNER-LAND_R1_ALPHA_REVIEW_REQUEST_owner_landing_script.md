# X-OWNER-LAND · R1 · REVIEW_REQUEST · The owner's landing script, before its first real use

## Prior record

**Searches run** (fresh public clone at `d60a714`; VM101 and ser8 were not read by Alpha):

- `grep -rnIiE "owner_land|landing script|git_push\.sh|push_to_claude|commit list|no-touch|hand-landed|dual-push|dual push|inbox"`
  over `docs/PROJECT_FILE_CATALOG.md` (0 files), `docs/TB_RULING_*` (4), `docs/PROPOSAL_*` (6),
  `docs/CHAPTER_*` (1), `docs/*_AUDIT.md` (0), `docs/TODO_CONSOLIDATED.md` (0),
  `docs/BACKLOG.md` (1), `docs/LEADS.md` (0), `docs/exchange/` (4), `CLAUDE.md`, `AGENTS.md`.
  Every hit in those scopes was read.
- The same pattern without the two dual-push terms over `docs/CLAUDE_CODE_REPORT_*` and
  `docs/CLAUDE_CODE_INSTRUCTIONS_*` (17 files). Read: the landing-script brief and report in
  full, and the binding addendum §B. The other hits were read at the matching lines: they
  are the no-touch rule repeated in the 2026-11-13 briefs and reports, and the August
  Defect A and pre-run documents, which use "no-touch" for their own lists.
- Full read: `scripts/owner_land.py` (731 lines), `docs/exchange/RELAY_PROTOCOL.md`,
  `CLAUDE.md`, `AGENTS.md`, `docs/TFM_SESSION_HANDOFF_20261113.md`. Read in part:
  `tests/test_owner_land.py:1-60` and its `live_state()` and `main()`.
- `git log --oneline -15`; `sha256sum` of the two code files; `sed -n '41p' .gitignore`;
  `git check-ignore -v data/owner_land/x`.

**Documents found that already decide or record the subject:**

| Document | What it says | Label |
|---|---|---|
| `CLAUDE.md` §1 rules 1–2; `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:140` | Agents never commit or push. Michael commits and dual-pushes, `origin` then `public` | ALREADY RECORDED. Not changed |
| `AGENTS.md` §2 table; completion plan `:141-142` | Tools are reviewed **after** commit (owner ruling, 2026-11-11) | ALREADY RECORDED. This review is after commit; the owner has asked for it before first use |
| Completion plan `:177` | "Owner commits … Batched: one commit list per wave" | ALREADY RECORDED as intent. Manifest mode is that list |
| `docs/PROPOSAL_FINDINGS_20261113_RESOLUTION_v1_1.md` §0 | The no-touch file list; "Landing: Michael commits and dual-pushes, between RANGE-MINER launches" | ALREADY RECORDED. Bears on Q6 and Q8 |
| `docs/CLAUDE_CODE_INSTRUCTIONS_BINDING_ADDENDUM_BETA_R1_20261113.md` §B | Adds `gate12_launch.sh` to the no-touch list | ALREADY RECORDED. Bears on Q6 and Q8 |
| `docs/exchange/RELAY_PROTOCOL.md` §2.3, §4, §6 | Nothing else writes to the tree while a relay runs; every commit stays with Michael; the ledger records the request and reply sha256 | ALREADY RECORDED. Bears on Q4 |
| `docs/TFM_SESSION_HANDOFF_20261113.md` §2 item 9, §6 items 4–6 | "Build the landing script first"; the inbox-branch idea is undecided; a reused download name and an untracked settings file both caused trouble on 2026-11-13 | ALREADY RECORDED |
| `docs/CLAUDE_CODE_INSTRUCTIONS_OWNER_LAND_SCRIPT_20261113.md`; `docs/CLAUDE_CODE_REPORT_OWNER_LAND.md` | The specification, and the build report with its own prior record, 33/33 result, open items and UNAVAILABLE list | ALREADY RECORDED (`5e100b3`, `240e907`) |
| `docs/TB_RULING_*` (4 hits), `docs/BACKLOG.md:374,388`, `docs/CHAPTER_CORRECTIONS_REPORT_20261111.md:764` | Older "Michael may commit and dual-push" wording; the dual-push command | ALREADY RECORDED. No bearing on the script |

**Labels for what this request raises:**

| Item | Label |
|---|---|
| Q1, the guards against the brief | Review of work ALREADY RECORDED in the report |
| Q2 unpushed or diverged `main`; Q3 `modify` over an uncommitted edit; Q4 exchange-mode hashes; Q5 ignored destinations | NEW. Not in the brief, the report or any document searched |
| Q6 the process watch list | The broad match is ALREADY RECORDED (report §7 item 3). The measured table below and the launch-harness gap are NEW |
| Q7 Codex environment names | NEW question. The guard itself is recorded (report §2) |
| Q8–Q10 | ALREADY RECORDED as open owner decisions (report §6, §7 item 1). Alpha's recommendations are stated; the owner has not decided |
| Q11 first real use | NEW. It changes no ruling |

**Not searched:** VM101 (including `.claude/settings.json`, which is not tracked), ser8, the
private repository, shell aliases, project files that are not in the repository.

---

```
id / round      X-OWNER-LAND / R1
type            REVIEW_REQUEST
from / to       Team Alpha (Claude)  ->  Team Beta (Codex)
date            2026-11-14
review class    Tool, AFTER-COMMIT (owner ruling 2026-11-11). Committed at 240e907 and never
                used. The owner has asked for your review before its first real use.
base_commit     HEAD at the time you read (the commit that adds this file, on top of d60a714)
files to read   scripts/owner_land.py        sha256 618c623cb1132e7f9c5dccc5b7769683559bc090e08ddf6576fbaeb0841e327d
                tests/test_owner_land.py     sha256 f6e24869a76f4711f234da8626e4305006e71f411738dc7bdd2f0188f29e9880
                docs/CLAUDE_CODE_REPORT_OWNER_LAND.md
                docs/CLAUDE_CODE_INSTRUCTIONS_OWNER_LAND_SCRIPT_20261113.md
```

Allowed answers per question: `NO OBJECTION` · `OBJECTION` (with `LIVE` or `THEORETICAL` and
one sentence why) · `SUGGESTION` · `NEEDS EVIDENCE`.

**Two limits on how you check this.**

- **Do not run `scripts/owner_land.py`, in any mode.** It commits and pushes (`AGENTS.md` §4.2).
- **Do not run the suite inside the repository.** It writes under `data/owner_land/` of the
  tree it sits in (`tests/test_owner_land.py:29`, `:46-47`), and that path is not ignored
  (`.gitignore:166-167` list two other `data/` folders). A run there creates files outside
  `from_beta/`. If you want to run it, a local clone under a temp directory outside the
  repository is the way that leaves this tree alone; the suite needs to sit in a git checkout
  (`live_state()`). `UNAVAILABLE` is an acceptable answer for the suite.

## 1. The facts

Measured by Alpha this session:

1. Both code files at `d60a714` have the sha256 the report gives (§2 there).
2. The suite passes on a fresh public clone at `d60a714`: `33/33 PASS`,
   `COMPLETION SENTINEL: PASS`, with Python 3.13.18 and git 2.43.0. The report's run was on
   VM101 with Python 3.10.12 and git 2.34.1. The clone's `git status` was empty afterwards.
3. `git add -- a.json` on a new file matched by a `*.json` ignore rule exits 1 (throwaway
   repository). `.gitignore:41` in this repository is `*.json`.
4. `classify_process` (`scripts/owner_land.py:387-424`), called directly on these argument
   lists. The function is pure; nothing else in the script ran.

   | Command line | Result |
   |---|---|
   | `python3 scripts/beta_relay.py --request x` | matched (relay) |
   | `python3 agents/watcher_agent.py --run-pipeline` · `python3 -u -m agents.watcher_agent …` | matched (WATCHER) |
   | `python3 window_optimizer.py --strategy bayesian` | matched (window optimizer) |
   | `python3 -u tests/test_s172_phase4_coordinator.py` | matched (coordinator) |
   | `pytest tests/test_s172_phase4_coordinator.py` · `python3 -m pytest …` · `python3 …/bin/pytest …` | **not matched** |
   | `/bin/bash ./gate12_launch.sh` · `bash scripts/launch_fleet_manual.sh` | **not matched** |
   | `bash python3_with_venv.sh agents/watcher_agent.py …` (the wrapper process itself) | not matched |
   | `python3 -u miner/range_miner_worker.py --gpu-id 0` | not matched |
   | `python3 chapter_13_orchestrator.py --daemon` | not matched |
   | `codex exec …` · the Claude Code process | not matched |

   The report's example in §7 item 3 (`pytest tests/test_s172_phase4_coordinator.py` refuses a
   landing) does not hold for the three `pytest` forms. It holds for the `python3 -u <suite>`
   form this project uses.

Read from the source, not executed:

5. Nothing compares local `main` with the two remotes before a landing. `check_remotes`
   (`:524-526`) tests only that the remote names exist. The pushes are
   `git push <remote> main` (`:657`), which send every local commit on `main`, not only the
   new one.
6. For a `modify` entry, `check_entries` requires the destination to be a tracked file
   (`:494-498`) and refuses if the incoming bytes equal HEAD (`:507-510`). It does not
   compare the copy in the tree with HEAD. `check_dirty` skips planned paths (`:438-442`).
   The copy at `:619` then overwrites the tree copy. The prior bytes are kept only for a
   rollback (`:636-643`).
7. Exchange mode lands the ledger, the deferred list and the thread's untracked files as they
   stand in the tree (`:294-303`). It reads three ledger columns: thread, round and verdict
   (`:93`, `:314-324`). It does not read the ledger's request and reply sha256 columns.
8. A failed `git add` (`:624-625`) is caught and rolled back with exit 5 (`:630-652`). That
   happens after the owner has typed `land`.
9. The agent-environment guard matches `CLAUDECODE`, `AI_AGENT`, and names starting
   `CLAUDE_CODE_` or `CODEX_` (`:97-98`, `:357-363`).
10. The hand-landed set is `GOVERNED_FILES` read from `scripts/gate12_parity_gate.py`, plus
    `agents/watcher_agent.py`, `gate12_launch.sh`, the parity gate itself, `miner/` and
    `agent_manifests/` (`:84-87`, `:461-476`). The no-touch list in the Findings proposal §0
    and addendum §B is longer.

## 2. Questions

**Q1 — The guards against the brief.** Do `all_checks` (`:542-554`), the confirmation
(`:586-593`), the second pass of the checks (`:595-600`) and the rollback (`:630-652`) do what
the brief's §1 lists, in every mode? Name any path by which a file reaches a commit or a push
without the plan having shown it.

**Q2 — Unpushed or diverged `main` (fact 5).** If local `main` already holds a commit that is
not on a remote, the landing pushes it too, and the printed plan says "one commit". Alpha's
position: add a refusal when HEAD differs from `origin/main` or `public/main` as last
fetched. Is that needed before first use?

**Q3 — `modify` over an uncommitted edit (fact 6).** If a planned `modify` destination has an
uncommitted change in the main tree, a successful landing replaces it without saying so.
Alpha's position: in manifest and worktree modes, refuse when the tree copy differs from
HEAD. Is that needed before first use?

**Q4 — Exchange mode and the ledger's hashes (fact 7).** The relay records the request and
reply sha256. The script does not check the files it lands against them, so a reply changed
after the relay ran would land as found. Alpha's position: verify both hashes for the round
being landed, and refuse on a mismatch. Is that needed before exchange mode is first used?

**Q5 — Ignored destinations (facts 3 and 8).** A `new` destination matched by an ignore rule
fails at `git add`, after the confirmation, and is rolled back. Alpha's position: check
`git check-ignore` with the other checks, so the refusal comes before the owner types
anything. Position?

**Q6 — The process watch list (fact 4).** The Findings proposal §0 places landings "between
RANGE-MINER launches". The launch harness and the fleet launcher are not matched, so a
landing can change HEAD after the clean-tree admission gate has passed and before WATCHER
starts. Alpha's position: add `gate12_launch.sh` and `launch_fleet_manual.sh` by name, and
correct the report's `pytest` example. Is the list otherwise sufficient?

**Q7 — Codex environment names (fact 9).** You are the only party who can say this. From
your own session, which environment variable **names** mark a Codex shell? Names only, no
values. Does the `CODEX_` prefix cover them?

**Q8 — The hand-landed set (fact 10).** Should the script refuse the full no-touch list
(Findings proposal §0 plus addendum §B) and not only the brief's shorter set? Alpha
recommends the full list. The owner has not decided.

**Q9 — The rule and the deny entry.** The report (§6) proposes one line for `CLAUDE.md` §1
forbidding any agent to run the script, and a Claude Code deny entry. Alpha recommends the
line as written and the entry `Bash(*scripts/owner_land.py*)`, so that agents can still run
the suite. The report's §7 item 2 states what the guards do not stop. Is that statement
complete?

**Q10 — The two old scripts.** `git_push.sh` and `push_to_claude.sh` run `git add -A` or
`git add .` and push one remote. Alpha recommends replacing each body with a one-line
refusal that names `scripts/owner_land.py`. The report recommends deleting them. Position?

**Q11 — First real use.** The report lists as never exercised: a real `ssh -t` session, the
real `input()` prompt, and pushes to the real remotes. The proposed first use is manifest
mode with two new document files under `docs/`. Is that acceptable as the first real use on
the script as committed? If not, which of Q2–Q6 must be fixed first?

**Q12 — Prior record.** Is there a governing document on any of Q1–Q11 that the section
above missed?

## 3. What Alpha did not verify

- Anything on VM101: the deny-list in `.claude/settings.json`; whether
  `.claude/settings.local.json` appears in `git status` there (`.gitignore` has no `.claude`
  entry; if it appears, every landing refuses with `DIRTY-PATH`).
- Facts 5 to 8 by execution. Each is a source read. The measurement that settles each is a
  gate in the suite's own style, in a throwaway repository.
- A real terminal, the real remotes, and `default_processes()` against live processes.
- The suite body beyond `:1-60`, `live_state()` and `main()`.

## 4. What follows

Alpha turns the accepted items into one brief for a Claude Code session. A change to the
script is landed by hand, because the script lands only the committed copy of itself
(`:529-539`). Q8 to Q10 are the owner's decisions; your positions go to him beside Alpha's.
