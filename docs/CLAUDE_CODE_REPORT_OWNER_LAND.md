# CLAUDE CODE REPORT — THE OWNER'S LANDING SCRIPT — 2026-11-13

**Brief:** `docs/CLAUDE_CODE_INSTRUCTIONS_OWNER_LAND_SCRIPT_20261113.md` · **Run on:** VM101 as
`michael` · **Worktree:** `/home/michael/tfm_worktrees/owner_land` (detached at `5e100b3`; the
brief names base `8a314e5`, and `5e100b3` is that commit plus the brief itself) · **Review
class:** tool, AFTER-COMMIT. It goes to Beta before its first real use. · Nothing committed or
pushed. The script was never run against the real repository.

## Prior record

**Searches run** (worktree at `5e100b3`, which holds the same `docs/` as the main checkout):

```
P='owner_land|landing script|git_push\.sh|push_to_claude|commit list|gpush|dual-push|dual push'
/bin/grep -rnIiE "$P" docs/PROJECT_FILE_CATALOG.md docs/TB_RULING_* docs/TB_RULING_REQUEST_* \
    docs/PROPOSAL_* docs/CHAPTER_* docs/*_AUDIT.md docs/TODO_CONSOLIDATED.md docs/BACKLOG.md \
    docs/LEADS.md docs/exchange/
```

176 files matched the `TB_RULING*/PROPOSAL*/CHAPTER*/*_AUDIT` globs, and the four named files all
exist. I also read `.claude/settings.json` (main checkout) for the deny-list, and
`docs/PROPOSAL_FINDINGS_20261113_RESOLUTION_v1_1.md` §0 after it came up in the search.

**Documents found:**

| Found | What it records | Label |
|---|---|---|
| `CLAUDE.md` §1 rules 1–2 | agents never commit or push; Michael dual-pushes `origin` then `public` | ALREADY RECORDED. Not changed; the script runs that rule for him |
| `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:140,177` | "Michael commits and dual-pushes"; "Batched: one commit list per wave" | ALREADY RECORDED as intent. The `manifest` mode is that commit list |
| `docs/PROPOSAL_FINDINGS_20261113_RESOLUTION_v1_1.md:33` | "Michael commits and dual-pushes, between RANGE-MINER launches" | ALREADY RECORDED. The script enforces it by refusing while a coordinator runs |
| same file, §0 "No-touch files" (line 28) | a wider no-touch list than the brief's hand-landed set (see §7 item 1) | ALREADY RECORDED. **Not applied.** It is the owner's call |
| `docs/CHAPTER_CORRECTIONS_REPORT_20261111.md:764` | `git push origin main && git push public main` | ALREADY RECORDED. Same order as the script |
| `docs/TB_RULING_*` (4 hits) | Beta "may commit and dual-push" wording in older rulings | ALREADY RECORDED. Historical wording, superseded by the 2026-11-11 owner rulings (`CLAUDE.md` header). No bearing on the script |
| `docs/BACKLOG.md:374,388` | commit + dual-push as a completion step for the skill file | ALREADY RECORDED. No bearing |
| `docs/exchange/RELAY_PROTOCOL.md` §2.3 | "No other session writes to the working tree while the relay runs" | ALREADY RECORDED. The script refuses while `beta_relay.py` runs |
| `.claude/settings.json` (main checkout) | denies `Bash(git commit:*)`, `Bash(git push:*)`; asks on `Bash(git:*)` | ALREADY RECORDED. It does not see `owner_land.py` (§6) |
| `git_push.sh`, `push_to_claude.sh` | `git add -A` / `git add .`, a single `git push` | ALREADY EXISTS (brief's Prior record). Recommendation in §5 |
| No hit for `owner_land` or `gpush` anywhere in scope | — | The script is **NEW** |

**Items this report raises:**

| Item | Label |
|---|---|
| `scripts/owner_land.py`, `tests/test_owner_land.py` | NEW (the brief authorises them) |
| The refusals the brief lists (§1 step 2) | NEW, as specified |
| The refusals I added beyond the brief (§2, table "Additions") | NEW. Listed separately so Beta and the owner can strike any of them |
| Remove `git_push.sh` and `push_to_claude.sh` (§5) | NEW recommendation. Owner decides |
| The `CLAUDE.md` line and the deny-list entry (§6) | NEW. The `CLAUDE.md` line adds a rule; it changes no ruling. Owner decides |
| Widening the hand-landed set to the §0 no-touch list (§7 item 1) | Would apply an ALREADY RECORDED list to a new tool; owner decides |

**Not searched:** the Zeus host, ser8 archives, the private repository, project files that are
not in the repository, shell aliases on VM101 or ser8 (the owner's `gpush` alias is outside the
repo). No SESSION_CHANGELOG was written: the brief allows only the three files below.

## 1. First step: no relay, coordinator, WATCHER or optimizer

Run at 2026-11-13T20:25:18-08:00 from the main checkout (HEAD `5e100b3`, `git status --porcelain`
empty):

```
--- pgrep -af beta_relay.py (instructed; may self-match)
23828 /bin/bash -c source /home/michael/.claude/shell-snapshots/... eval '... pgrep -af beta_relay.py; ...'
rc=0
--- ps self-safe scan  ([b]eta_relay\.py|[c]oordinator|[w]atcher_agent|[w]indow_optimizer)
  23828   22696 /bin/bash -c source ... (the same shell)
--- my shell pid 23828
```

**The only hit, in both scans, was PID 23828: the shell running the scan** (the self-match from
`~/CLAUDE.md` hard rule 8). No relay, coordinator, WATCHER or window optimizer was running. The
same trap is the reason the script matches the *program a process runs*, not text in its
command line (gate `G-PROC-DECOYS`).

## 2. What was built

| File | sha256 | Lines |
|---|---|---|
| `scripts/owner_land.py` | `618c623cb1132e7f9c5dccc5b7769683559bc090e08ddf6576fbaeb0841e327d` | 731 |
| `tests/test_owner_land.py` | `f6e24869a76f4711f234da8626e4305006e71f411738dc7bdd2f0188f29e9880` | 603 |
| `docs/CLAUDE_CODE_REPORT_OWNER_LAND.md` | this file | — |

All three are new files. No existing file was edited. Standard library only (Python 3.10.12, git
2.34.1 on VM101).

**The brief's checks, as built** (`scripts/owner_land.py`, function `all_checks`). Every one
prints its own tagged `REFUSED` line, all failures are printed together, and the exit is `3`
with nothing changed:

| Brief item | Tag | How |
|---|---|---|
| stdin not a terminal | `NOT-A-TERMINAL` | `sys.stdin.isatty()` (injectable) |
| branch not `main` · linked worktree | `BRANCH` · `LINKED-WORKTREE` | `symbolic-ref`; `--absolute-git-dir` ≠ `--git-common-dir`. Also `NOT-TOPLEVEL` if not run from the repo root |
| relay / coordinator / WATCHER / window optimizer running | `PROCESS-RUNNING` | `/proc/*/cmdline`: the executable, or the script or `-m` module a Python interpreter runs, named `beta_relay.py`, `watcher_agent.py`, `window_optimizer*.py`, `*coordinator*.py`. Its own PID is skipped |
| dirty path not in the plan | `DIRTY-PATH` | `git status --porcelain -z --untracked-files=all`, so each untracked file is named, not its directory |
| destination hand-landed | `GOVERNED` | `GOVERNED_FILES` is **read from `scripts/gate12_parity_gate.py` at run time** with `ast` (no copy, no import), plus `miner/`, `agent_manifests/`, `agents/watcher_agent.py`, `gate12_launch.sh`. If the tuple cannot be read, it refuses |
| `new` exists · `modify` missing | `NEW-EXISTS` · `MODIFY-MISSING` | in the tree **and** at HEAD |
| sha256 mismatch · source missing | `SHA-MISMATCH` · `SOURCE-MISSING` | before anything is copied |
| empty message | `MESSAGE` | also refuses a multi-line message |
| `origin` or `public` missing | `REMOTE` | `git remote` |
| typed word ≠ `land` | exit `4` | `answer.strip() != "land"`. Case-sensitive; EOF aborts. No flag skips it |

**Additions beyond the brief** (NEW; each can be struck without touching the rest):

| Addition | Why |
|---|---|
| `AGENT-ENVIRONMENT`: refuse if `CLAUDECODE`, `AI_AGENT`, `CLAUDE_CODE_*` or `CODEX_*` is set | The brief says it must be impossible for an agent to use. This session's shell carries all four kinds of variable (§3, `REAL-CLI`) |
| `SELF`: refuse unless the running file is `<repo>/scripts/owner_land.py` and equals HEAD's blob | An agent could edit a copy, or the working file, to drop a guard. Only the committed script lands. Side effect: it can never land a change to itself; that, like its first landing, is done by hand |
| All checks run **again** after `land` is typed | The tree can change while the owner reads the plan (`G-RECHECK`) |
| `UNCHANGED`: refuse a `modify` whose bytes equal HEAD | Avoids an empty or partial commit |
| Path safety: no absolute path, `..`, `.git/`, or inbox file outside the inbox; no duplicate destinations | A manifest is input from elsewhere |
| `scripts/gate12_parity_gate.py` itself is hand-landed | It defines the governed set; a change to it changes the gate |
| Exchange mode refuses if the thread's last ledger row is already at HEAD, and takes only files named `<thread>_…` (so `X-0042` does not take `X-00420_…`) | Lands only a new round |
| If `git push origin main` fails, exit `6` and print the command for both pushes | The brief covers only the second push failing |
| Rollback restores each copied file's **exact prior bytes and mode**, removes `new` files and the directories created for them, and unstages the plan | Equal to "restore from HEAD" when the file was clean; it also keeps an owner's edit that was already in the plan |

Exit codes: `0` landed on both remotes · `2` usage · `3` refused · `4` aborted at the confirmation ·
`5` failed before the commit, rolled back · `6` committed, a push failed.

## 3. Gates — results

Command: `source ~/venvs/torch/bin/activate && PYTHONDONTWRITEBYTECODE=1 python3 -u tests/test_owner_land.py | tee data/owner_land_logs/run2.log`
(worktree-local; the logs are not for commit). **Result: `33/33 PASS`, `COMPLETION SENTINEL: PASS`.**
Each gate builds a throwaway repo with bare `origin` and `public` under the worktree's
`data/owner_land/` (removed on a pass), with `GIT_CONFIG_GLOBAL` pointed at a test config. Real
`scripts/gate12_parity_gate.py` is copied into each repo, so `G-GOVERNED` exercises the real
`GOVERNED_FILES`. Failures come from real git hooks.

**Null result first:** run 1 was `32/33` with `COMPLETION SENTINEL: FAIL`. `G-CONFIRM` crashed on
a test-side bug (the fake reader called the `EOFError` class instead of raising it). I fixed the
fake, not the script. Run 2 is the passing run. M-CONFIRM was re-earned in run 2 without that
crash (§4).

| Gate | Brief row | Expectation | Result |
|---|---|---|---|
| G-CLEAN-MANIFEST | clean control, manifest | one commit, both remotes at it, tree clean, commit = plan | PASS — `README.md` (modify) + `docs/new_dir/NOTES.md` (new) |
| G-CLEAN-WORKTREE | clean control, worktree | same, message from `-m` | PASS |
| G-CLEAN-EXCHANGE | clean control, exchange | same; message from the ledger | PASS — `exchange: X-0042 R1 (REVIEW RECEIVED)`, 4 files incl. new `DEFERRED.md` |
| G-NOTTY | not a terminal | refused, unchanged, never asks | PASS — rc 3, asked 0 |
| G-CONFIRM | typed ≠ `land` | `yes`, `Land`, empty, `land now`, EOF: aborted, unchanged | PASS — rc 4 ×5 |
| G-DIRTY | unexpected dirty path | refused, names `stray_notes.txt` | PASS |
| G-DIRTY-TRACKED | (extra) | a modified tracked file not in the plan is named | PASS |
| G-SHA | sha256 mismatch | refused, nothing copied (`docs/new_dir` absent) | PASS |
| G-NEW-EXISTS · G-MODIFY-MISSING | new exists · modify missing | refused | PASS · PASS |
| G-SOURCE-MISSING | source missing | refused, manifest and worktree mode | PASS |
| G-GOVERNED | governed set | `miner/…`, `sieve_gpu_worker.py`, `prng_registry.py`, `agents/watcher_agent.py`, `gate12_launch.sh`, `agent_manifests/x.json`, `scripts/gate12_parity_gate.py`: each named | PASS |
| G-LINKED-WT | linked worktree | run from a linked worktree on `main`: refused | PASS |
| G-BRANCH | branch not `main` | refused, names `feature` | PASS |
| G-PROC | relay running (injected) | `beta_relay.py`, coordinator by path and by `-m`, `watcher_agent.py`, `window_optimizer.py`, `./coordinator.py`: each refused | PASS |
| G-PROC-DECOYS | (extra) | `bash -c "pgrep -af beta_relay.py"`, `vim agents/watcher_agent.py`, `less coordinator.py`, own PID: **not** refused, lands | PASS |
| G-PUSH2 | second push fails | rc 6, prints `cd <repo> && git push public main`, origin advanced, public not, local commit kept | PASS |
| G-ROLLBACK | failure after copy, before commit | `pre-commit` hook exits 1: rc 5, tree/index/HEAD/remotes identical to before, new dir gone | PASS |
| G-ROLLBACK-EXCH | (extra) | same in exchange mode; Beta's files are not deleted | PASS |
| G-ADD-ONLY | `git add` gets only planned paths | recorded git calls: one `add -- <path>` per planned path, nothing else | PASS — manifest and exchange |
| G-MESSAGE | (empty message) | `-m "  "` and a manifest with no `message:` line: refused | PASS |
| G-REMOTE | (remote missing) | `public` removed: refused | PASS |
| G-AGENT-ENV | (addition) | each of `CLAUDECODE`, `CLAUDE_CODE_ENTRYPOINT`, `AI_AGENT`, `CODEX_SANDBOX`: refused | PASS |
| G-EXCHANGE-GUARDS | (addition) | no new round: refused; another thread's file left over: refused, named | PASS |
| G-RECHECK | (addition) | a file appears while the prompt waits: refused after `land`, HEAD and remotes unchanged | PASS |
| G-SELF | (addition) | running copy not the repo's: refused | PASS |
| G-PATH-SAFETY | (addition) | `../escape.md`, `.git/config`, inbox `../A`: refused | PASS |
| HYG-LIVE | (hygiene) | main checkout and worktree: HEAD, status, all refs unchanged by the suite | PASS |

**The real CLI, from this agent shell** (`data/owner_land_logs/mutant_detail.log`), against a
throwaway fixture repo, not the real one: exit 3, fixture unchanged, three refusals:
`NOT-A-TERMINAL`, `AGENT-ENVIRONMENT` (it lists 9 Claude Code variables), and `SELF`.

## 4. Mutant evidence

The suite applies each mutant to the source text (the anchor must occur exactly once), loads it,
and requires the named gates to go red. All 5 were killed. Detail (`mutant_detail.log`):

| Mutant | Change | Killing gate | What the mutant did |
|---|---|---|---|
| M-TTY | `check_terminal` removed | G-NOTTY | **landed** with no terminal: rc 0, tree changed |
| M-CONFIRM | `if answer… != "land"` → `if False` | G-CONFIRM | **landed** on `yes`: rc 0 |
| M-DIRTY | `check_dirty` removed | G-DIRTY, G-DIRTY-TRACKED | G-DIRTY: **landed** beside an untracked stray (rc 0). G-DIRTY-TRACKED: still refused, but by `GOVERNED` (the fixture's edit made `GOVERNED_FILES` unreadable), not by `DIRTY-PATH`. It was red because the named refusal was missing; it was not a landing |
| M-ADD-A | `add -- <path>` → `add -A` | G-ADD-ONLY | issued `['add','-A']` per entry. In a clean tree the commit is the same, so only the recorded-call gate sees it |
| M-PUSH2 | push loop limited to the first remote | G-CLEAN-MANIFEST, G-PUSH2 | reported success (rc 0) with `public` not advanced, and printed no finishing command |

## 5. Usage, from ser8

The script requires a terminal on the VM101 side, so `-t` is required. Paths are absolute
(`CLAUDE.md` §2).

**manifest** — stage Alpha's files and the manifest in the inbox first (`mkdir -p` before
`scp`, `CLAUDE.md` §2):

```
ssh michael@192.168.3.177 'mkdir -p /home/michael/tfm_inbox'
scp <files> MANIFEST michael@192.168.3.177:/home/michael/tfm_inbox/
ssh -t michael@192.168.3.177 'cd /home/michael/distributed_prng_analysis && python3 scripts/owner_land.py --manifest'
```

`MANIFEST` layout (`new|modify` with or without brackets; `#` lines ignored; names without spaces):

```
message: docs: one-line message
<sha256>  NOTES.md  ->  docs/NOTES.md  new
<sha256>  README.md  ->  README.md  modify
```

Use `--manifest /path/to/OTHER` or `--inbox /path` to land from elsewhere. The script does not
empty the inbox; clear it after a landing.

**worktree:**

```
ssh -t michael@192.168.3.177 'cd /home/michael/distributed_prng_analysis && python3 scripts/owner_land.py --worktree owner_land scripts/x.py tests/test_x.py -m "tools: x"'
```

**exchange:**

```
ssh -t michael@192.168.3.177 'cd /home/michael/distributed_prng_analysis && python3 scripts/owner_land.py --exchange X-RUN4-RANGE'
```

**Before the first real use:** the script cannot land itself (the brief says so, and `SELF`
requires the committed copy). Michael lands the three files by hand, as he did tonight.

## 6. `git_push.sh`, `push_to_claude.sh`, and the rule

**Recommendation: delete both** (`git rm git_push.sh push_to_claude.sh`) in the hand landing that
lands this script, or in the first landing after it. Both use `git add -A` / `git add .`, push
one remote, and have no guards (brief Prior record). `push_to_claude.sh` commits without a prompt.
A shell with an agent in it can run either of them, and neither is covered by the deny-list. A
softer alternative: reduce each to a one-line `echo "use scripts/owner_land.py"; exit 1`. Owner
decides. I edited neither file.

**One-line rule for `CLAUDE.md` §1** (owner adds it; it adds a rule and changes no ruling):

> **No agent runs `scripts/owner_land.py`, or anything that invokes it, in any mode or against any repository. It is the owner's landing tool. Agents run only `tests/test_owner_land.py`.**

**Matching deny-list entry** (suggested for the main checkout's `.claude/settings.json`, beside
the existing `Bash(git commit:*)` / `Bash(git push:*)`): `"Bash(*owner_land.py*)"`. That also
denies `python3 -u tests/test_owner_land.py`. The suite would then need either an explicit
allow for that exact command, or the owner to run it.

## 7. Open items, limits, UNAVAILABLE

1. **Owner decision: the hand-landed set.** The brief's set (`GOVERNED_FILES`, `miner/`,
   `agents/watcher_agent.py`, `gate12_launch.sh`, `agent_manifests/`) is narrower than the
   §0 "No-touch files" list in `docs/PROPOSAL_FINDINGS_20261113_RESOLUTION_v1_1.md:28`. That list
   adds `window_optimizer*.py`, `coordinator.py`, `scripts/gate12_*`,
   `scripts/launch_fleet_manual.sh`, `tests/gate_s172_prod_shape.py`, `utils/run_finalizer.py`,
   `utils/canonical_records.py`, `utils/seed_coverage_ledger.py` and `distributed_config.json`.
   The no-touch list governs agent edits, not landings, so I built to the brief. If the owner
   wants it here too, the change is one tuple (`HAND_LANDED_FILES` / `HAND_LANDED_DIRS`).
2. **Limits of the agent guard (measured, stated plainly).** The terminal check, the
   environment check and the typed word stop an agent that runs the script the ordinary way.
   That covers this session (§3, `REAL-CLI`). An agent that deliberately builds a pseudo-terminal
   (`script -c …`), scrubs its environment (`env -i`) and writes `land` to it would get past all
   three. The barrier against that is the `CLAUDE.md` rule and the deny-list entry in §6, not the
   script.
3. **Process match is fail-closed and broad.** Any running Python script with `coordinator` in
   its name refuses a landing, and that includes `pytest tests/test_s172_phase4_coordinator.py`.
   The relay protocol and the Findings plan both want landings kept apart from live runs, so I
   treated a false refusal as the cheaper error. The port-5700 check in the Findings plan
   (`:32`) is not part of the script.
4. **UNAVAILABLE** (not exercised in this session):
   - a real `ssh -t` session from ser8, and a human typing at the real `input()` prompt. Only
     the injected reader was exercised; the real `isatty()` was exercised only in its `False`
     branch;
   - pushes to the real GitHub `origin` and `public` (only local bare remotes);
   - `default_processes()` against a live `beta_relay.py` or coordinator. It ran for real once
     (the `REAL-CLI` run) and found nothing, because nothing was running. The matching logic
     itself is covered by `G-PROC` / `G-PROC-DECOYS` on injected lists;
   - the Beta review (AFTER-COMMIT class, due before first real use);
   - fallback parity on `.127` (not in this brief).

## 8. Stop gate

Suite `33/33 PASS`, `COMPLETION SENTINEL: PASS` (run 2). Report written. Stopping here. The
three files are in `/home/michael/tfm_worktrees/owner_land/` for Michael to land by hand.
