# READY · Identity base pinned to 827f4b4 in both suites after landing (2026-11-24): screen 15/15 and 13/13, Brief II 18/18 and 12/12

- **Task:** `logs/e2e_session/TASK_SCREEN_TEST_BASE_PIN_20261124.txt` (lead, 17:48).
- **Follow-up:** lead answers 18:18. Q1: pin the Brief II suite too, and add a non-vacuity check. Q2: stay detached at main.

## Update after lead answers 18:18 (current state; supersedes the open questions below)
**Landing list, test files only:**

| File | Change |
|---|---|
| `tests/test_step1_screen.py` | `PRE_CHANGE_BASE = 827f4b4…` replaces `merge-base HEAD main` in `base_commit()`; the docstring is updated. Its existing `DEFOBJ-nonvacuous` check is kept |
| `tests/test_window_anchor_brief2.py` | `PRE_CHANGE_BASE = 827f4b4…` and `_identity_base()` replace `merge-base HEAD main`. `_head_source(rel, check_id)` now refuses a base equal to the live file, under check ids `DEFOBJ-nonvacuous` (gate `G-DEFAULT-OBJ`) and `DEFSEARCH-nonvacuous` (gate `G-DEFAULT-SEARCH`). The gate docstrings are updated |

- **Diff of both files:** `data/screen_base_pin/r2/two_test_files.diff`, sha256 `6955d411…00ed`, 36 insertions and 17 deletions.
- `TFM_IDENTITY_BASE` still overrides the base in both suites.
- No production code changed.

**Results on main `abdf9f7` plus the two test diffs**, run sequentially on `taskset -c 6,7`, `nice -n 15`. Tree state is in `data/screen_base_pin/r2/tree_state.txt`: `HEAD abdf9f7…`, only the two test files modified.

| Suite | Base | Gates | Mutants | Sentinel | Output (`data/screen_base_pin/r2/`) |
|---|---|---|---|---|---|
| `tests/test_step1_screen.py` | 827f4b4 (pinned) | **15/15** | **13/13** | PASS | `test_step1_screen.out` |
| `tests/test_window_anchor_brief2.py` | 827f4b4 (pinned) | **18/18** | **12/12** | PASS | `test_window_anchor_brief2.out` |
| `tests/test_window_anchor_brief2.py`, negative check: `TFM_IDENTITY_BASE=abdf9f7` | abdf9f7 (the landed code) | 16/18 | 11/12 | **FAIL, as intended** | `test_window_anchor_brief2_base_abdf9f7_negative.out` |

**The negative row shows the new check bites.** Pointed at the landed code, both identity gates now refuse instead of passing vacuously:
> `FAIL G-DEFAULT-OBJ [DEFOBJ-nonvacuous] window_optimizer.py: base abdf9f7… equals the live file, so identity would prove nothing`
> `FAIL G-DEFAULT-SEARCH [DEFSEARCH-nonvacuous] utils/step1_search_ledger.py: base abdf9f7… equals the live file, so identity would prove nothing`

**State left:**
- The worktree stays detached at `abdf9f7` (Q2), and branch `anchor-brief2` is unchanged at `280fdfd`.
- No optuna studies were left in the worktree.
- No commit, no push.

---
*The first round follows (17:48 task), kept as recorded. Its Q1 and Q2 are answered above.*
- **Evidence:** `data/screen_base_pin/` in this worktree. It holds the diff, `run_suites.sh`, `tree_state.txt` and three suite outputs.

## Prior record
**Searches run**, from the main tree:
- `/bin/grep -rln` for `TFM_IDENTITY_BASE`, `merge-base HEAD main`, `nonvacuous`, `PRE_CHANGE_BASE`, `identity base` and `base pin`;
- over `docs/PROJECT_FILE_CATALOG.md`, `docs/TB_RULING_*`, `docs/TB_RULING_REQUEST_*`, `docs/PROPOSAL_*`, `docs/CHAPTER_*`, `docs/*_AUDIT.md`, `docs/TODO_CONSOLIDATED.md`, `docs/BACKLOG.md`, `docs/LEADS.md`, `docs/exchange/`, `docs/drafts/anchor_v1/` and `tests/`.

**Found:**
- `docs/drafts/anchor_v1/DESIGN_CHEAP_SCREEN_20261123.md:341,364`. Here the identity base already moved once, from `HEAD` to `merge-base HEAD main`, after WIP snapshot `b78a15c` made `HEAD` the changed code. The same line records that only the screen suite's `G-DEFAULT-OBJ` checks that the base differs from the live file.
- `tests/test_step1_screen.py:80-87` and `tests/test_window_anchor_brief2.py:310-314` are the two places where the base is resolved.
- No ruling, proposal, backlog or LEADS entry covers the identity base.

**Labels for what this note raises:**

| Item | Label |
|---|---|
| Pinning the screen suite's base to a named commit | NEW; the follow-on to `DESIGN_CHEAP_SCREEN_20261123.md:364` |
| The Brief II suite's identity gates being vacuous on main | NEW |
| The `merge-base` mechanism itself | ALREADY RECORDED (`DESIGN_CHEAP_SCREEN_20261123.md:364`) |

**Not searched:** the host, ser8 archives, the private repository, project files outside the repository.

## Plain summary
- **The screen suite passes on landed main with the base pinned: GATES 15/15, MUTANTS 13/13, sentinel PASS.**
  - `G-DEFAULT-OBJ` now compares against `827f4b4` and passes, including `DEFOBJ-nonvacuous`.
  - Mutant S10, which was not executed before, is now applied, executed and detected (`test_step1_screen.out:208`).
- **The change is to the test file only**, one hunk plus the docstring line. `PRE_CHANGE_BASE = 827f4b4fb5cf…` replaces `merge-base HEAD main`, and `TFM_IDENTITY_BASE` still overrides it. The base is resolved with `rev-parse --verify`, so an unknown commit fails loudly.
  - `827f4b4` has the same `.py` files as `41ad509`: `git diff --quiet 41ad509 827f4b4 -- '*.py'` is clean.
  - Diff: `data/screen_base_pin/test_step1_screen.diff`, sha256 `9852cb36…7dd4`.
- **`test_window_anchor_brief2.py` has the same dependence and does *not* avoid it.** Its 18/18 on main is vacuous for the claim its gates make.
  - `_head_source` (`:310-314`) uses `merge-base HEAD main`. On main that is `abdf9f7`, the changed code.
  - `G-DEFAULT-OBJ` and `G-DEFAULT-SEARCH` therefore compare the landed code with itself. The suite has no non-vacuity check.
  - Its identity mutant M7 is still detected, but only because the mutant edits the working file and the base is the committed file. On main the gate proves "the working tree equals the commit", not "default-off equals pre-change".
- **Measured: with the base pointed at `827f4b4`, the Brief II suite still passes 18/18, 12/12** (`TFM_IDENTITY_BASE=827f4b4`). Default-off identity against the pre-change code does hold.
  - I did not edit that file; the task named the screen suite only.
  - The same one-hunk pin would make its gates non-vacuous by default. A non-vacuity check like the screen suite's would also stop this class from recurring.

## Runs
- **Tree:** this worktree, detached at main `abdf9f7`, plus the one test diff. `tree_state.txt`: `HEAD abdf9f7…`, only `M tests/test_step1_screen.py`.
- **How they ran:** sequentially, on `taskset -c 6,7`, `nice -n 15`.

| Suite | Base | Gates | Mutants | Output |
|---|---|---|---|---|
| `tests/test_step1_screen.py` (pinned) | 827f4b4 | **15/15** | **13/13** | `test_step1_screen.out` |
| `tests/test_window_anchor_brief2.py` as landed | merge-base = abdf9f7 (vacuous) | 18/18 | 12/12 | `test_window_anchor_brief2.out` |
| `tests/test_window_anchor_brief2.py`, `TFM_IDENTITY_BASE=827f4b4` | 827f4b4 | **18/18** | **12/12** | `test_window_anchor_brief2_base827f4b4.out` |

Before the pin, the lead's run on main gave 14/15 and 12/13 (`~/tfm_worktrees/_review/main_final_test_step1_screen.py.out:22,35,52`).

## Worktree move (disclosed)
- **The move:** the worktree was on branch `anchor-brief2` at `280fdfd` (WIP snapshot), with `utils/step1_screen.py` modified. That modification was byte-identical to `abdf9f7`'s file; a copy is in `data/screen_base_pin/step1_screen.py.worktree_copy`.
  - I restored the file to the branch version and ran `git checkout --detach abdf9f7`.
  - The branch still points at `280fdfd`. Drafts that exist only on the branch, such as `NOTE_SINGLE_CARD_SIEVE_20261124.md`, the PREP notes and the first two READY timing notes, are not in the working tree now, but nothing is lost.
  - `git checkout anchor-brief2` brings them back.
- **What was not done:** no commit, no push, no production code change. No optuna studies were left in this worktree (0 files).

## Questions
- **Q1.** Lead: should the Brief II suite get the same pin, and a non-vacuity check, as a test-only change?
- **Q2.** Lead: should the worktree stay detached at `abdf9f7`, or go back to branch `anchor-brief2`?
