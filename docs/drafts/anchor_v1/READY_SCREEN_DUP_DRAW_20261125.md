# READY · cheap screen: refuse a repeated (date, session) (grader R26) (2026-11-25)

Task: `~/distributed_prng_analysis/logs/e2e_session/TASK_SCREEN_DUP_DRAW_20261125.txt` (lead 14:45).

**Work tree:** `~/tfm_worktrees/screen-dupdraw`, detached at main `b2b76fb`. Nothing committed, pushed or landed. Nothing written in `~/distributed_prng_analysis`.

## Prior record

**Searches run** (on `b2b76fb` with `git grep`; files ignored by `.gitignore` are not searched; `logs/e2e_session/` read in the main tree with `/bin/grep`, read only):

```
P='repeated (date, session)\|repeated.row\|repeated-row\|R26\|dup.draw\|DUP_DRAW\|duplicate.*(date, session)\|(date, session).*unique'
git grep -n -e "$P" b2b76fb -- docs/PROJECT_FILE_CATALOG.md 'docs/TB_RULING_*' 'docs/PROPOSAL_*' 'docs/CHAPTER_*' \
  'docs/*_AUDIT.md' docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/LEADS.md docs/exchange/ docs/DAILY3_CONSUMER_CONTRACT_v1.md
git grep -ln -e "$P" b2b76fb -- docs/drafts
git grep -n -e 'R26\|repeated\|dup' b2b76fb -- docs/drafts/anchor_v1/DESIGN_CHEAP_SCREEN_20261123.md docs/drafts/loop_v1/READY_LAND_WF_VOTE_20261125.md
git grep -n screen b2b76fb -- docs/LEADS.md docs/BACKLOG.md docs/TODO_CONSOLIDATED.md   (filtered for dup|repeat: none)
/bin/grep -rln 'R26\|repeated (date\|DUP_DRAW\|dup-draw\|dupdraw\|repeated-row\|repeated row' ~/distributed_prng_analysis/logs/e2e_session/
```

**Documents found that decide or record the subject:**
- **The rule:** `docs/DAILY3_CONSUMER_CONTRACT_v1.md:444` ("`(date, session)` is unique across the array. No duplicates, ever"); `:95`; `:419` (other consumers rely on it).
- **The objection the grader fix answered:** `docs/exchange/from_beta/X-E2E-GRADER-LANDED_R1_BETA_REVIEW.md` Q2 (`:44`, "Why LIVE": the scraper once recorded 721 duplicates before dedup).
- **The grader's R26:** `scripts/diagnostics/e2e_grader.py:149-155` (header), `:334-336` (check in `load_dataset`); gate `tests/test_e2e_grader.py:327-348` (`test_repeated_draw`).
- **This gap, reported and not fixed:** `docs/drafts/e2e_v1/READY_GRADER_DUP_DRAW_20261124.md`, section "Found while doing this", first bullet ("The cheap screen has no repeated-row check of its own … the lead's call").
- **The task that ordered it:** `logs/e2e_session/TASK_SCREEN_DUP_DRAW_20261125.txt` (it cites "Handoff section 0 item 2"; I did not find that handoff among the searched files — see Not searched).

**Labels:**

| Item | Label |
|---|---|
| Uniqueness of `(date, session)` | ALREADY RECORDED (contract `:444`) |
| The screen lacks the check | ALREADY RECORDED (`READY_GRADER_DUP_DRAW_20261124.md`, "Found while doing this") |
| The screen's check and its gate/mutants | NEW (this change) |
| CHANGES A RULING | none |

**Not searched:** the host, the ser8 archives, the private repository, project files outside the repository, other worktrees; the handoff the task names (not located in `docs/` or `logs/e2e_session/`).

## Plain summary

- **The change.** `build_screen_plan` now refuses a dataset in which one `(date, session)` occurs at two row indices. It raises `Step1ScreenError("screen: dataset rows I and J: repeated (date, session) (...)")` — the grader's wording with the module's `screen:` prefix — at `utils/step1_screen.py:440-445`, right after `json.load` and before anything else is derived. Key = `(str(date), str(session))`, the same as the grader's (`e2e_grader.py:333`). 6 lines added, none removed.
- **Before any write or sieve work.** `build_screen_plan` writes nothing. Its one caller, `window_optimizer_integration_final.py:3699-3702`, runs before the search (`optimizer.optimize`, `:3745`). The screen needs `--derived-anchor-domain`, which refuses `n_parallel != 1` (`:2847-2848`, `window_optimizer.py:1938-1939`), so the NP2 study set-up block (`:3195`) does not run on a screen path.
- **Default unchanged.** With the default objective (`bidirectional_count`) this module's plan is never built (`window_optimizer_integration_final.py:2873`, `:3695`). With a screen objective on a clean dataset, every plan and score output is byte-identical to `b2b76fb` (table below).
- **Tests.** `tests/test_step1_screen.py` gains gate **G-DUP-DRAW** (`:508-536`, registered `:720`), following the grader's `test_repeated_draw`:
  - DUP-FIXTURE: rows 250 and 251 share a date with different sessions;
  - DUP-SESSION: the clean fixture (every day has both sessions) still builds a plan;
  - DUP-REFUSE: row 250 copied to 251 is refused, the message naming "dataset rows 250 and 251";
  - DUP-NOOUT: the refusal leaves the scratch dir as it was and creates no Optuna study.

  Two new mutants (`:634-639`), both **killed and credited**: **S14** removes the check (DUP-REFUSE); **S15** keys it on date alone (DUP-SESSION). No existing line of the test changed (0 lines removed; `GRADER`, `GRADER_SHA`, `HARNESS`, `PRE_CHANGE_BASE` untouched).

## Proof on a fresh copy of main

- **Setup:** `git clone --no-checkout ~/distributed_prng_analysis ~/tfm_worktrees/_check_screen_dupdraw`, `git checkout --detach b2b76fb`, then **only** `utils/step1_screen.py` and `tests/test_step1_screen.py` copied in (`git status`: those 2 files modified). A clone, not `git worktree add`, so no bookkeeping was written under `~/distributed_prng_analysis/.git` while the other agent's relay runs there.
- **How it ran:** one at a time, foreground, `PYTHONDONTWRITEBYTECODE=1 taskset -c 0-3 nice -n 10 python3 -u -B`, venv `~/venvs/torch`, no card.
- **Afterwards:** only the 2 copied files tracked-modified; no `__pycache__`; `~/distributed_prng_analysis/optuna_studies/` listing identical before and after. The check tree was then removed (`rm -rf`, after confirming its HEAD was `b2b76fb`).

| Check | Result | Log (`~/tfm_worktrees/screen-dupdraw/logs/screen_dupdraw/`) | sha256 |
|---|---|---|---|
| `tests/test_step1_screen.py` | **GATES 16/16** (15 + G-DUP-DRAW), **MUTANTS 15/15 DETECTED** (13 + S14, S15), **COMPLETION SENTINEL: PASS** | `check_test_step1_screen.log` (`:194`, `:213-214`) | `23821056…` |
| `tests/test_window_anchor_brief2.py` | **GATES 18/18, MUTANTS 12/12, COMPLETION SENTINEL: PASS** | `check_test_window_anchor_brief2.log` | `ee0ceeb3…` |
| `tests/test_e2e_grader.py` | **37/37** | `check_test_e2e_grader.log` | `e07ed825…` |
| Default equivalence vs `b2b76fb` | **PASS**: `screen_hits` 58,400 bytes and `screen_rank` 59,935 bytes identical (sha256 `7087f7eb…`, `60421832…` both sides) | `check_default_equiv.log` | `cbb17c8f…` |

**Default equivalence check** (`default_equiv_check.py.txt`, sha256 `5f0db621…`, kept as `.txt` so it is not an untracked `.py`). It loads `utils/step1_screen.py` from `git show b2b76fb:` and the tree's copy as two modules and, on the suite's clean 2,000-row fixture, for both screen objectives, compares canonical JSON of: `run_record()`, `tag()`, sessions by row, combined indices per session option, draws, `selection_end_max`, confirmation blocks, `anchor_max` for 12 (option, width) pairs, and `score()` for 3 record sets × 3 windows. Only the wall-clock `seconds` field is dropped. **Non-vacuity:** the two sources differ (`3afd3d1c…` vs `9586f7ce…`); the good records score 200 hits across the two objectives, so the compared outputs are not empty; on the repeated-row dataset the base **accepts** (plan built) and the tree **refuses** ("dataset rows 250 and 251: repeated (date, session) ('2000-05-05', 'evening')").

The Brief II log contains Optuna "Trial 0 failed" warnings from its refusal gates; they are expected, as recorded in `READY_GRADER_DUP_DRAW_20261124.md`.

## Found while doing this (not changed)

- The grader suite leaves `data/e2e_grader_selftest/run_<ts>/` behind in the tree it runs in (seen in the check clone before removal). Pre-existing; the clone is gone.
- The contract's other consumers (`DAILY3_CONSUMER_CONTRACT_v1.md:419`: `evaluate_pools.py:23`, `backtest_pools.py:47`) still assume uniqueness without checking. Not examined.

## Landing list (repo-relative, against main `b2b76fb`)

| # | Path | N/M | sha256 at `b2b76fb` → this change |
|---|---|---|---|
| 1 | `utils/step1_screen.py` | M | `3afd3d1c54f4…` → **`9586f7cecf1a73040e318422a5509102637730161331c4e558c22cc4acdfc81d`** |
| 2 | `tests/test_step1_screen.py` | M | `31ae4b3450d2…` → **`c7d93a6431bb86be46a10f696db4842ce5bc0b0a93858e66d9f386428247a243`** |
| 3 | `docs/drafts/anchor_v1/READY_SCREEN_DUP_DRAW_20261125.md` | N | this note |

- `git diff --stat`: 2 files, 46 insertions, 0 deletions.
- No `.json` file; nothing needs force-adding. No pin changes: the grader is untouched, so `GRADER_SHA` stays `3ab8ce2a…`.
- The logs and the equivalence script are in the ignored `logs/screen_dupdraw/` of this worktree. Copy them next to this note if they are to be kept.
