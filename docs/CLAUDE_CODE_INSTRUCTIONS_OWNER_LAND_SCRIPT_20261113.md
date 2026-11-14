# CLAUDE CODE INSTRUCTIONS — THE OWNER'S LANDING SCRIPT — 2026-11-13

## Prior record

**Searches run** (public mirror at `8a314e5`; VM101 and ser8 were not read by Alpha):

- `grep -lE "git (commit|push)" *.sh scripts/*.sh automation/*`
- `grep -rnE "push_to_claude|gpush|commit list" README.md CLAUDE.md AGENTS.md docs/exchange/RELAY_PROTOCOL.md docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md docs/BACKLOG.md docs/TODO_CONSOLIDATED.md`
- read in full: `git_push.sh`, `push_to_claude.sh`, `CLAUDE.md` §1 rules 1–2, `docs/exchange/RELAY_PROTOCOL.md` §3–§5
- `git log --oneline -8` for the landings made by hand on 2026-11-13

**Documents and files found:**

| Found | What it is | Label |
|---|---|---|
| `git_push.sh` (repo root, 2025-12-09) | prompts for a message, then `git add -A`, `git commit`, one `git push` | ALREADY EXISTS. Breaks two current rules: never `git add -A`; always push both remotes |
| `push_to_claude.sh` (repo root, 2025-11-29) | `git add .`, automatic message, one `git push` | ALREADY EXISTS. Same two breaks |
| `CLAUDE.md` §1 rules 1–2 | agents never commit or push; Michael commits and dual-pushes `origin` then `public` | ALREADY RECORDED. This brief does not change it |
| `docs/exchange/RELAY_PROTOCOL.md` §4, §5 | every commit and push stays with Michael; nothing writes to the tree while a relay runs | ALREADY RECORDED |
| `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:177` | "Owner commits … Batched: one commit list per wave" | ALREADY RECORDED as intent; never built |
| Commits `ea85436`, `2930a37`, `29a6cb9`, `8a314e5` | four landings typed by hand tonight, each with a copy, a guard on `git status`, explicit `git add`, dual push | The behaviour this script reproduces |

**What is NEW:** one script that does the landing the owner did by hand, with the guards
built in. It replaces nothing in the rules.

**Not searched:** shell aliases and scripts outside the repository on VM101 or ser8 (the
owner has a `gpush` alias; it is interactive-only and is not in the repository).

---

**From:** Team Alpha · **Owner:** Michael · **Run on:** VM101, as `michael`, venv
`source ~/venvs/torch/bin/activate` · **Model:** Opus · **Base commit:** `8a314e5`
**Review class:** tool, AFTER-COMMIT. It goes to Beta before its first real use.

**Why.** The owner makes every commit. Tonight each landing cost several commands and a
visual check. This script turns one landing into one command with a printed plan and one
typed confirmation. It must be impossible for an agent to use it, because it is a path to
`git commit` and `git push` that the Claude Code deny-list does not see.

## 0. Rules for this session

- **You do not commit or push. You never run the script against the real repository.**
  Every test builds its own throwaway repository with two bare remotes in a temp directory.
- **Work in a worktree:** `git worktree add --detach /home/michael/tfm_worktrees/owner_land HEAD`.
  Write only inside it. Create no branch, tag or commit.
- **New files only:**
  - `scripts/owner_land.py`
  - `tests/test_owner_land.py`
  - `docs/CLAUDE_CODE_REPORT_OWNER_LAND.md` (opens with `## Prior record`, per `CLAUDE.md`)
  - scratch under `data/owner_land/` (not for commit)
- **No existing file is edited.** `git_push.sh` and `push_to_claude.sh` are left as they are;
  say in the report what should happen to them.
- **First step:** check that no relay is running (`pgrep -af beta_relay.py`) and no
  coordinator, WATCHER or window optimizer; record the output. If a relay is running, wait.
- No fleet, no GPU, no rig SSH. Nothing is installed.
- Standard library only. Run Python with `PYTHONDONTWRITEBYTECODE=1`.

## 1. What the script does

`scripts/owner_land.py` lands a **named list of files** as **one commit** on `main` and pushes
it to **both** remotes. It has three ways to be given the list.

**Mode `manifest` (files Alpha delivered).** The owner copies Alpha's files and one manifest
into an inbox directory outside the repository (default `/home/michael/tfm_inbox/`). The
manifest is plain text:

```
message: <one-line commit message>
<sha256>  <filename in the inbox>  ->  <destination path in the repo>  [new|modify]
...
```

The script verifies each file's sha256, copies it to its destination, and lands the list.

**Mode `worktree` (an agent's output).** `--worktree <name> <path> [<path> ...] -m <message>`
copies the named paths from `/home/michael/tfm_worktrees/<name>/` to the same paths in the
main tree, and lands them.

**Mode `exchange` (a relay round).** `--exchange <thread-id>` lands whatever the relay left
for that thread: `docs/exchange/LEDGER.md`, `docs/exchange/DEFERRED.md` if modified, new
files in `docs/exchange/from_beta/` and `docs/exchange/to_beta/` whose names start with the
thread id. The message is built from the ledger's last row for that thread (thread, round,
verdict).

**In every mode, in this order:**

1. Print the plan: each file, `new` or `modify`, its sha256, and the commit message.
2. **Refuse, changing nothing, if any of these holds** (each with its own message):
   - standard input is not a terminal;
   - the current branch is not `main`, or the directory is a linked worktree, not the main
     checkout;
   - a `beta_relay.py` process, a coordinator, WATCHER or window optimizer is running;
   - `git status --porcelain` shows any path that is not in the plan;
   - a destination is in `GOVERNED_FILES` (`scripts/gate12_parity_gate.py`), is under
     `miner/`, or is `agents/watcher_agent.py`, `gate12_launch.sh` or anything in
     `agent_manifests/` (the owner lands those by hand);
   - a destination marked `new` already exists, or one marked `modify` does not;
   - a sha256 does not match (manifest mode), or a source file is missing;
   - the commit message is empty;
   - either remote `origin` or `public` is not configured.
3. Ask the owner to type the word `land`. Anything else aborts and changes nothing.
   **There is no flag that skips this.**
4. Copy the files, `git add` **each path by name** (never `-A`, never `.`), commit.
5. `git push origin main`, then `git push public main`. If the first succeeds and the second
   fails, exit non-zero and print the exact command that finishes the job.
6. Print `git status --porcelain` and `git log --oneline -1`.

If any step after the confirmation fails before the commit exists, restore every copied
file to its prior state (remove `new` files, restore `modify` files from `HEAD`) and say so.

## 2. Tests

`tests/test_owner_land.py`, in the project's house style: named gates, a clean control,
mutants that must turn a named gate red, and a completion sentinel. Each test builds a
temp repository with two bare remotes. The terminal check and the typed word are supplied
through an injectable reader, so no test needs a real terminal.

At least these gates:

| Gate | Expectation |
|---|---|
| clean control, each of the three modes | one commit, both remotes at that commit, tree clean |
| not a terminal | refused, nothing changed |
| typed anything but `land` | aborted, nothing changed |
| unexpected dirty path in the tree | refused, names the path |
| sha256 mismatch | refused, nothing copied |
| `new` destination exists · `modify` destination missing | refused |
| destination in the governed set | refused |
| linked worktree · branch not `main` | refused |
| relay process running (injected process list) | refused |
| second push fails | non-zero exit, prints the finishing command, first remote advanced, local commit kept |
| failure after copy, before commit | files restored, tree as before |
| `git add` receives only the planned paths | asserted on the recorded git calls |

Mutants, each killed by a named gate because the mutant lands something it should refuse:
the terminal check removed; the confirmation removed; the dirty-path check removed;
explicit paths replaced by `-A`; the second push removed.

## 3. Report

`docs/CLAUDE_CODE_REPORT_OWNER_LAND.md`: the `## Prior record` section; the first-step
output; the gate table with results; the mutant evidence; exact usage lines for the three
modes, written as commands the owner runs from ser8 with `ssh -t`; what should happen to
`git_push.sh` and `push_to_claude.sh`; the one-line rule to add to `CLAUDE.md` forbidding any
agent to run the script; everything `UNAVAILABLE`.

## 4. Stop gate

Stop when the suite passes and the report is written. Tell Michael the report path. He
lands the three files by hand, since the script cannot land itself before it exists in
`main`.
