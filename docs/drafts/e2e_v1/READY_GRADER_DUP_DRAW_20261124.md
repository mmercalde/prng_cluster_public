# READY · grader R26: refuse a repeated (date, session) (2026-11-24)

Task: `~/distributed_prng_analysis/logs/e2e_session/TASK_GRADER_DUP_DRAW_20261124.txt` (lead 19:00; owner yes 18:57).

**Work tree:** `~/tfm_worktrees/grader-dup-draw`, a detached worktree at main `bce6407`, made for this task. Nothing committed, pushed or landed.

## Prior record

**Searches run** (on `main` = `bce6407`, with `git grep`; files ignored by `.gitignore` are not searched):

```
git grep -n -e <P> main -- docs/PROJECT_FILE_CATALOG.md 'docs/TB_RULING_*' 'docs/PROPOSAL_*' 'docs/CHAPTER_*' \
  'docs/*_AUDIT.md' docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/LEADS.md docs/exchange/ docs/DAILY3_CONSUMER_CONTRACT_v1.md
  P = 'duplicate.*(date, session)\|(date, session).*unique\|unique.*date.*session' | X-E2E-GRADER-LANDED | e2e_grader | c7fe48bf
grep -rn c7fe48bf  <main-equal work tree>  and  ~/tfm_worktrees/e2e-map (excluding data/)
grep -rn 'grader_sha256\|grader-sha256\|GRADER_SHA\|e2e_grader' utils tests scripts docs/drafts/anchor_v1
```

**Documents found that decide or record the subject:**
- **The objection:** `docs/exchange/from_beta/X-E2E-GRADER-LANDED_R1_BETA_REVIEW.md`, Q2, OBJECTION LIVE (Positions table; "Q2 — Refusals"), read in full. Ledger row at `docs/exchange/LEDGER.md:58`.
- **The uniqueness rule:**
  - `docs/DAILY3_CONSUMER_CONTRACT_v1.md:444` ("`(date, session)` is unique across the array. No duplicates, ever");
  - `:95`, verified unique across three files;
  - `:419`, two other consumers rely on it: `evaluate_pools.py:23`, `backtest_pools.py:47`.
- **Run H's own check:** `docs/exchange/to_beta/X-E2E-GRADER_R1_ALPHA_REVIEW_REQUEST_grader_first_form.md:120` records 18,068 rows and no duplicates.
- **The pin:** `docs/drafts/anchor_v1/DESIGN_CHEAP_SCREEN_20261123.md:85` (its prior record) and `:371` (TODO-SCREEN-1, "Repin when it lands").

**Labels:**

| Item | Label |
|---|---|
| R26 refusal | ALREADY RECORDED as a requirement (contract `:444`; Beta Q2). Its enforcement in the grader is NEW |
| Repinning the screen test to the landed path | NEW (the test read the grader from the `e2e-grader` worktree, `tests/test_step1_screen.py:53` at `bce6407`) |
| The cheap screen has no repeated-row check of its own | NEW, reported, not fixed |
| CHANGES A RULING | none |

**Not searched:**
- the host, the ser8 archives, the private repository, project files outside the repository;
- worktrees other than `e2e-map`; the repinning section below covers `e2e-map`.

## Plain summary

- **The change.** The grader now refuses a dataset in which one `(date, session)` occurs at two row indices.
  - It raises `GraderError("dataset rows I and J: repeated (date, session) (...)")` in `load_dataset` (`scripts/diagnostics/e2e_grader.py:311-338`, check at `:334`). `main` turns that error into exit 2 with "GRADER ERROR (nothing written)" (`:1394-1396` at `bce6407`).
  - `load_dataset` runs after the out-dir is checked but before `write_outputs` creates it (`:1175` vs `:1077` at `bce6407`), so **nothing is written**.
  - The rule is written into the header as **R26** (`:149-155`).
  - Nothing else changed: 12 lines added, none removed.
- **No result so far is affected.** Beta checked that run H and the current dataset each have 18,068 unique pairs. On such a dataset the new check never fires.
- **Tests and mutants.** `tests/test_e2e_grader.py` gains the gate `test_repeated_draw` (`:327-348`), which follows Beta's construction:
  - row 250 is copied to index 251;
  - each index gets a pool holding the outcome, bound to the bad snapshot's digest;
  - the run must exit 2 and name both rows (A-R26-REFUSE);
  - no out-dir may be created (A-R26-NOOUT);
  - two sessions on the same date must still pass (A-R26-SESSION).

  Two new mutants are both **killed and credited**:
  - M51 removes the check (A-R26-REFUSE);
  - M52 keys the check on date alone (A-R26-SESSION).
- **The reference equivalence gate** (`test_counted_default_equivalence`, `:723-…`) still runs both cases against the frozen `a4d96fe7` grader byte for byte. Counted-default equivalence now means two things:
  - **on any dataset whose `(date, session)` pairs are unique, the outputs equal `a4d96fe7`'s,** apart from timestamp, provenance and the recorded mode. The gate now asserts that its own cases have unique pairs (A-EQUIV-DOMAIN), so it cannot pass on a case outside that domain;
  - **on a dataset with a repeat, `a4d96fe7` grades both rows (exit 0) and the current grader refuses (exit 2, no out-dir).** This is asserted as A-EQUIV-R26, so the one intended difference is measured, not assumed.

  The fixture `tests/fixtures/e2e_grader_a4d96fe7.py.txt` is unchanged (sha checked by A-EQUIV-REF). R26's header text records the same statement.
- **Sha256 of `scripts/diagnostics/e2e_grader.py`:**
  - old: `c7fe48bf102daa42b284713f4b2c2508d7f21c9613a01f063be443c881fd9ace`
  - **new: `3ab8ce2a5330fe36ae638e8bf2915e23bdce08bcefc0221eb47e21177fc8f341`**

## Proof on a fresh tree equal to main

- **Setup:** `git worktree add --detach ~/tfm_worktrees/_landing_check_dupdraw main` (= `bce6407`), then the **5 code and doc files** of the landing list copied in and nothing else. This note was written afterwards; no test reads it.
- **How it ran:** sequentially under `taskset -c 0-5 nice -n 10`, with `python3 -B` and `PYTHONDONTWRITEBYTECODE=1`, venv `~/venvs/torch`, no card (`CUDA_VISIBLE_DEVICES` unset; the screen and Brief II suites set it to `""` themselves).
- **Cleanup:** the worktree was removed afterwards with `git worktree remove --force`, and `git worktree list` no longer shows it. The add and the remove wrote git bookkeeping under `~/distributed_prng_analysis/.git/worktrees/`.
- **Afterwards:**
  - the check tree had no `__pycache__`, and its only tracked changes were the 5 copied files;
  - no new Optuna study appeared in `~/distributed_prng_analysis/optuna_studies/` (`find -newer` on the list file was empty).

| Suite | Result | Log (in `~/tfm_worktrees/e2e-grader/logs/e2e_grader/`) | sha256 |
|---|---|---|---|
| `tests/test_e2e_grader.py` | **37/37** (36 before, plus `test_repeated_draw`) | `dupdraw_check_test_e2e_grader.log` | `96163f5b…` |
| `tests/test_compare_settings.py` | **9/9** | `dupdraw_check_test_compare_settings.log` | `0f932a30…` |
| `tests/test_switch_check.py` | **12/12** | `dupdraw_check_test_switch_check.log` | `5e3a83eb…` |
| `tests/test_registry_repair.py` | **5/5** | `dupdraw_check_test_registry_repair.log` | `03024c13…` |
| `tests/mutate_e2e_grader.py` (all targets) | **89/89 killed and credited** (87 before, plus M51 and M52); baseline gates pass: True; 96.7 s | `dupdraw_check_mutate_e2e_grader.log` | `9c6fcea3…` |
| `tests/test_step1_screen.py` | **GATES 15/15, MUTANTS 13/13, COMPLETION SENTINEL: PASS**; G-GRADER-PIN passes against the new pin | `dupdraw_check_test_step1_screen.log` | `08310489…` |
| `tests/test_window_anchor_brief2.py` | **GATES 18/18, MUTANTS 12/12, COMPLETION SENTINEL: PASS** | `dupdraw_check_test_window_anchor_brief2.log` | `dfb015f1…` |

Copied-file list: `~/tfm_worktrees/e2e-grader/logs/e2e_grader/dupdraw_landing_list_20261124.txt` (`31503a20…`). These logs sit in the ignored `logs/` of the `e2e-grader` worktree. Copy them next to this note if they are to be kept.

The Brief II log contains Optuna "Trial 0 failed" warnings. They come from its refusal gates (`WindowAnchorDomainError`, window outside its bounds) and are expected; the suite's verdict is the GATES line.

## Repinning: every place that pins or reads `c7fe48bf…`

**On main (updated in this change):**

| File:line | What | Done |
|---|---|---|
| `scripts/diagnostics/e2e_grader.py` | the pinned bytes | changed (R26) |
| `tests/test_step1_screen.py:53-54` | `GRADER` was `~/tfm_worktrees/e2e-grader/scripts/diagnostics/e2e_grader.py`, **another worktree, not the repo**; `GRADER_SHA` was `c7fe48bf…` | `GRADER` now reads `ROOT/scripts/diagnostics/e2e_grader.py` (the landed file); `GRADER_SHA` = `3ab8ce2a…` |
| `docs/drafts/anchor_v1/DESIGN_CHEAP_SCREEN_20261123.md:371` (TODO-SCREEN-1) | "Repin when it lands" | dated update naming the landing (`76234eb`), R26, the new sha and this note |

**On main, reading the hash without stating it (no edit needed):**
- `utils/step1_screen.py:118-135` `load_grader` compares a screen config's `grader_sha256` to the file. It has no built-in hash; `SCREEN_DEFAULTS` holds `None` (`:64-65`). **Any screen config written for `c7fe48bf…` must name `3ab8ce2a…`**; `load_grader` refuses the old one.
- There are no screen-config example files on main (`grep -rln grader_sha256 docs` finds only the two design `.md` files).
- The signatures `load_grader` checks (`GRADER_FUNCS`, `:88-92`) are unchanged.
- `scripts/diagnostics/compare_settings.py:363`, `:456` takes `--grader-sha256` from its caller.
- `tests/test_compare_settings.py:152` computes the hash of the file under test.

**On main, stating the old hash as history (not edited):**
- `DESIGN_CHEAP_SCREEN_20261123.md:85`: that design's prior record.
- `docs/drafts/loop_v1/READY_LOOP_GRADE_20261123.md:54`, `:147`, `:290`.
- `docs/drafts/e2e_v1/READY_GRADER_LANDING_20261124.md:14`, `:20`, `:38`, `:56`, `:102`.
- `docs/exchange/to_beta/X-E2E-GRADER-LANDED_R1_ALPHA_REVIEW_REQUEST_grader_as_landed.md:7`.
- `docs/exchange/from_beta/X-E2E-GRADER-LANDED_R1_BETA_REVIEW.md:63`: a Beta file, which a session never edits.

**`~/tfm_worktrees/e2e-map`, the unlanded walk-forward driver (`953bcf4` WIP; not edited, reported only):**

| File:line | Now | Needs |
|---|---|---|
| `docs/drafts/loop_v1/runH_commands/00_base.sh:21` | `G=c7fe48bf…` | `G=3ab8ce2a5330fe36ae638e8bf2915e23bdce08bcefc0221eb47e21177fc8f341` |
| `docs/drafts/loop_v1/runH_commands/01_sieve_on.sh:21` | same | same |
| `docs/drafts/loop_v1/runH_commands/02_target_hits.sh:21` | same | same |
| `docs/drafts/loop_v1/runH_commands/03_rankers.sh:21` | same | same |
| `scripts/diagnostics/e2e_walk_forward.py:476-477` | `--grader` default `~/tfm_worktrees/e2e-grader/scripts/diagnostics/e2e_grader.py` | the landed path `~/distributed_prng_analysis/scripts/diagnostics/e2e_grader.py`, or pass `--grader` explicitly in the four scripts above, which do not pass it today |
| `scripts/diagnostics/e2e_walk_forward_selftest.py:65` | `GRADER` = the same `e2e-grader` worktree path; it snapshots that file (`:246-250`) | the landed path |

**Why the path matters as much as the hash.** The `e2e-grader` worktree still holds `c7fe48bf…`; it was not changed by this task. If only the four `G=` lines change, the driver checks `3ab8ce2a…` against the old worktree file and refuses (`e2e_walk_forward.py:537-541`). That fails closed. If only the path changes, it refuses the new file against the old `G`. Both must change together.

The driver's `--help` capability checks (`:542-550`) still pass: R26 adds no flag.

The driver's READY notes there (`READY_WF_SMOKE_20261124.md:37`, `READY_WF_TARGET_HITS_BLOCK0_20261124.md:61`, `READY_WF_SIEVE_ON_BLOCK0_20261124.md:31`, `READY_LOOP_WF_20261123.md:188`, `:300`) and its logs record the old pin as history.

## Found while doing this (not changed)

- **The cheap screen has no repeated-row check of its own.** `utils/step1_screen.py:438-442` loads the dataset with `json.load` and builds draws and sessions without going through the grader's `load_dataset`. It uses only `pb_upper_tail`, `grade_draw`, `validate_pool_values`, `draw_ranks` and `rank_summary` (`:88-92`), so **R26 does not reach the screen**. A repeated row in a screen's dataset would be graded twice in its selection and confirmation blocks.
  - The screen is development evidence only (design `:85`).
  - Adding the same check there is a separate, small change and is the lead's call. This task allowed no other behaviour change.
- **The contract's other consumers** (`DAILY3_CONSUMER_CONTRACT_v1.md:419`) also assume uniqueness. Not examined.
- **The stray bytecode in the `e2e-grader` worktree** (`scripts/diagnostics/__pycache__/e2e_grader.cpython-310.pyc`, 2026-11-23 20:44, reported earlier) is probably explained by the old `tests/test_step1_screen.py:53`. That test loaded the grader from that worktree with `spec_from_file_location` and `exec_module`, which writes bytecode unless it is disabled. [UNVERIFIED: I did not find which run wrote it.] The new line 53 points at the test's own tree.

## Landing list (repo-relative, against main `bce6407`)

| # | Path | N/M | sha256 at `bce6407` → this change |
|---|---|---|---|
| 1 | `scripts/diagnostics/e2e_grader.py` | M | `c7fe48bf102d…` → **`3ab8ce2a5330…`** |
| 2 | `tests/test_e2e_grader.py` | M | `29cc7f5df96a…` → `64aae2c4d542…` |
| 3 | `tests/mutate_e2e_grader.py` | M | `712b681b3a06…` → `7694e8f8e6f6…` |
| 4 | `tests/test_step1_screen.py` | M | `e271747b0a59…` → `418ff454fa02…` |
| 5 | `docs/drafts/anchor_v1/DESIGN_CHEAP_SCREEN_20261123.md` | M | `056b7c9073db…` → `1bf04a4c71d2…` |
| 6 | `docs/drafts/e2e_v1/READY_GRADER_DUP_DRAW_20261124.md` | N | this note |

- No `.json` file is involved, so nothing needs force-adding.
- The fixture `tests/fixtures/e2e_grader_a4d96fe7.py.txt` is unchanged.
- The files are in `~/tfm_worktrees/grader-dup-draw` (`git diff --stat`: 5 files, 63 insertions, 4 deletions). That worktree is mine and can be removed after landing.
