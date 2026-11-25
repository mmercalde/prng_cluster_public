# READY · landing e2e-vote (vote harness) and e2e-map (walk-forward driver) onto main `fe5ac3b` (2026-11-25)

- **Task:** `~/distributed_prng_analysis/logs/e2e_session/TASK_LAND_WF_VOTE_20261125.txt` (lead), plus `TASK_LAND_WF_VOTE_20261125_ADDENDUM_1.txt` (owner, 06:35: variant B, with a refusal requirement).
- **Author:** Claude (Team Alpha).
- **Edited worktrees:** `~/tfm_worktrees/e2e-map` and `~/tfm_worktrees/e2e-vote`.
- **Status:** prepared and proven only. Nothing is committed, pushed or landed, and nothing has gone to Beta. No card was used (`CUDA_VISIBLE_DEVICES=""`).
- **Evidence:** every log, probe record and list cited below is in `~/tfm_worktrees/e2e-map/data/landing_wfvote_20261125/` (ignored `data/`). Copy it next to this note if it is to be kept.

## Prior record

**Searches run** (on `main` = `fe5ac3b`, `git grep -l`; files ignored by `.gitignore` are not searched):

```
S = docs/PROJECT_FILE_CATALOG.md docs/TB_RULING_* docs/TB_RULING_REQUEST_* docs/PROPOSAL_* docs/CHAPTER_*
    docs/*_AUDIT.md docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/LEADS.md docs/exchange/
git grep -l -e P fe5ac3b -- $S   for P in: e2e_walk_forward | e2e_vote_harness | 'walk-forward\|walk_forward' |
    'LL-1\|LL-2\|LL-3' | feature_manifest | 'per_seed_sieve_features\|per-seed-sieve' |
    'X-E2E-VOTE\|X-E2E-WF\|X-E2E-MAP' | 'capability check' | 3ab8ce2a
git grep -n 'drafts/loop_v1\|e2e-vote\|e2e-map' fe5ac3b -- docs/LEADS.md docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/PROJECT_FILE_CATALOG.md
```

**Documents found, each read at the hit:**
- `docs/exchange/from_beta/X-E2E-GRADER-LANDED_R1_BETA_REVIEW.md:39`: cites `e2e-vote:scripts/diagnostics/e2e_vote_harness.py` as evidence for the grader review. It decides nothing about landing the vote harness.
- `docs/TODO_CONSOLIDATED.md:225`: "Walk-forward simulation | TM S148→S154 | OWNER". This is an old owner TODO and does not concern this driver.
- `docs/CHAPTER_7_*`, `CHAPTER_13_AUDIT.md`, `PROPOSAL_STEP1_SEARCH_RECORD_*`, `PROPOSAL_WATCHER_KPI_*`: generic "walk-forward" wording only. None concerns these branches.
- The `LL-*` hits (`TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md:94`, `X-MINER-STALL_R1_BETA_REVIEW.md:124`) are "WALL-2" mutant text, a false match.
- **No ruling, proposal or ledger row authorises, schedules or reviews the landing of `e2e-vote` or `e2e-map`.**
- The pattern followed comes from the main-tree landing notes:
  - `docs/drafts/e2e_v1/READY_GRADER_LANDING_20261124.md`: fixtures moved to `tests/fixtures/`, plus the force-add warning at `:111`;
  - `docs/drafts/e2e_v1/READY_GRADER_DUP_DRAW_20261124.md`: the repin table, and its list of `e2e-map` sites, which this task carries out.

**Labels:**

| Item | Label |
|---|---|
| Repin the grader `c7fe48bf` → `3ab8ce2a` in `e2e-map` | ALREADY RECORDED as needed (`READY_GRADER_DUP_DRAW_20261124.md`, "e2e-map" table); done here |
| Repo-relative vote and grader paths (no `~/tfm_worktrees/...`) | NEW |
| Frozen vote harness revisions rev4 and rev5 moved from `data/` to `tests/fixtures/` | NEW (same practice as `READY_GRADER_LANDING_20261124.md`) |
| Pipeline-switch capability check and refusal (owner ADDENDUM_1) | NEW |
| Step 3 harness accepts main's `score_survivors` signature | NEW (it refused main, see §1) |
| Landing variant B | owner direction, ADDENDUM_1; no ruling changed |
| CHANGES A RULING | none |

**Not searched:**
- the host, the ser8 archives, the private repository, project files outside the repository;
- the lead's handoff. The task cites "handoff section 0 item 1" and "lesson 3", and I found no handoff dated 2026-11-24 or 2026-11-25 on disk [UNVERIFIED]. I took "non-vacuity" to mean the pattern of `db741b8` (Brief II suite): show that the same comparison detects a real difference.

---

## A4. ADDENDUM 4 (lead, 08:55): Beta's two LIVE objections fixed; variant B re-proven with all 13 suites

- **Task:** `~/distributed_prng_analysis/logs/e2e_session/TASK_LAND_WF_VOTE_20261125_ADDENDUM_4.txt`.
- **Review fixed:** `docs/exchange/from_beta/X-E2E-WF-VOTE_R1_BETA_REVIEW.md`. The file is Beta's record; I quote only its position labels.
- **No second Beta round. Nothing committed, pushed or landed. Variant A not touched.**

### A4.1 Beta's objections and what was done

| Beta item | Severity (Beta) | Done | Gate |
|---|---|---|---|
| Q1, the wrapper: `01_sieve_on.sh` wrote a log and a `.time` file and printed `started` while its child refused | LIVE | **Fixed (Fix 1).** All four `runH_commands/0*.sh` build one `ARGS` array and run `python3 -u -B scripts/diagnostics/e2e_walk_forward.py "${ARGS[@]}" --check-only` in the **foreground** first (`01_sieve_on.sh:46`). On a non-zero exit they print `PREFLIGHT REFUSED (exit N): nothing launched; no log, no .time file, no out-dir written` and exit N (`:48-52`). Only after that do they `mkdir -p logs data` and `nohup` the same `ARGS` (`:53-56`). New driver flag `--check-only` (`e2e_walk_forward.py:501`, handled at `:1473`) runs every refusal the driver has, then exits 0 having written nothing | **W11**, 6 gates: the real wrapper in a tree whose Step 3 files lack LL-2 returns 2 with no log, no `.time`, no out-dir, no `logs/` and no launch. Non-vacuity: the same wrapper with LL-2 files passes the preflight and reaches the launch. 00, 02 and 03 reach the launch on the files without LL-2 |
| Q1, abbreviations: `--per-seed-sieve-f on` passed the driver and failed late in Step 3 | (inside Q1, LIVE) | **Fixed (Fix 2).** `allow_abbrev=False` in the Step 3 harness parser (`e2e_step3_harness.py:573`) and the shim parser (`e2e_step5_shim.py:118`). The driver preflight refuses any passthrough token that is an abbreviation of an option the receiving parser declares, and any undeclared option. The parser's options are read as source (`e2e_pipeline_caps.parser_options`, `:167`; `check_passthrough`, `:180`). Main's optimizer parser still accepts abbreviations (a pipeline file, not in B); the driver guard covers what reaches it | **W12**: driver refuses `--per-seed-sieve-f on` (exit 2, names it, nothing written); harness rejects it (`unrecognized arguments`, nothing created); shim rejects `--target`; mutant `passthrough_abbrev_allowed` caught |
| Q2, passthrough overrides: a trailing `--survivors=PATH` or `--k-folds` in `--step5-extra-args` won at the optimizer while the record said otherwise | LIVE (contested by me in §8; the owner rules) | **Fixed (Fix 3)** regardless of the label. The driver-set lists `STEP3_DRIVER_SET`, `STEP5_DRIVER_SET` and `SHIM_SET` (`e2e_walk_forward.py:93-95`) hold the options the driver and shim build. `check_passthroughs` (`:550`, called at `:607`, after the switch check so a missing LL switch is still named as such) refuses any of them in `--opt VALUE` or `--opt=VALUE` form in `--step3-extra-args`, `--step5-extra-args` or `--nn-args`, with exit 2 naming the option, before anything is written. `own_options_check` (`:569`) fails a run if the driver ever builds an option missing from its list, so the lists cannot drift. **Defence in depth:** the shim refuses `--survivors=…`, an abbreviation of `--survivors`, or a repeated `--survivors` before it creates its capture dir (`e2e_step5_shim.py:132-140`) | **W12**: refusals of `--survivors=PATH`, `--k-folds 2`, `--k-folds=2`, `--trials=3` via `--nn-args`, `--out-dir` via Step 3 and an unknown option. **Beta's regression gate:** source rows carry `holdout_quality 0.045` and `holdout_hits 2`. Through the real shim and **main's own optimizer parser body** (extracted from `main()` up to `parse_args`, run in a stub), the optimizer reads the shim's derived file (target 2) with the declared 3 folds, and a trailing `--survivors=SRC` is refused by the shim. Mutants `passthrough_eq_form_ignored`, `passthrough_driver_set_ignored` and shim `survivors_eq_allowed` are each caught; under the last one the optimizer reads the source file (0.045) |
| Q2: the snapshot vote record | — | Beta verified it (§8); nothing to do | — |
| Q3 | NO OBJECTION | — | — |
| Finding 1: the prior record left out the task logs | — | **Corrected** in A4.4 below | — |

### A4.2 Variant B re-proof, all 13 suites

- **Setup:** fresh `git worktree add --detach ~/tfm_worktrees/_landing_check_wfvote main` (`fe5ac3b`). Only the 33 list-B files were copied in (`list_B_r3.txt`, the same paths as `list_B.txt` `16413bf8…`; copy record `copied_B_r3.txt` `67fb6af2…`).
- **How the suites ran:** one at a time under `taskset -c 0-5 nice -n 10`, with `python3 -u -B`, `PYTHONDONTWRITEBYTECODE=1` and `CUDA_VISIBLE_DEVICES=""`. The runner was in the background and I polled it in the foreground.

| Suite | Round 3 (now) | log sha256 | Round 2 / round 1 (history) |
|---|---|---|---|
| `tests/test_e2e_vote_harness.py` | gates 15/15, mutants 17/17 | `209ca152…` | 15/15 17/17 · 14/14 16/16 |
| `tests/test_e2e_vote_ll5.py` | 3/3, 3/3 | `823e5d45…` | same |
| `tests/test_e2e_vote_ordering.py` | 6/6, 3/3 | `0c207ff5…` | same |
| `tests/test_e2e_miss_report.py` | 7/7, 5/5 | `9ddb70cc…` | same |
| `tests/test_step1_screen.py` | GATES 15/15, MUTANTS 13/13, SENTINEL PASS | `3fb4548b…` | same |
| `tests/test_e2e_grader.py` | 37/37 | `7df7f3dd…` | not re-run in round 2 · 37/37 |
| `tests/test_compare_settings.py` | 9/9 | `7c3d966f…` | not re-run · 9/9 |
| `tests/test_switch_check.py` | 12/12 | `f6efcbd1…` | not re-run · 12/12 |
| `tests/test_registry_repair.py` | 5/5 | `442017d5…` | not re-run · 5/5 |
| `tests/mutate_e2e_grader.py` | 89/89 killed and credited | `6b1b1bf5…` | not re-run · 89/89 |
| `tests/test_window_anchor_brief2.py` | GATES 18/18, MUTANTS 12/12, PASS | `d15afce1…` | not re-run · same |
| `scripts/diagnostics/e2e_walk_forward_selftest.py` | **99/99** (77 before + 6 W11 + 16 W12), 748 s | `6c550517…` | 77/77 · 77/77 |
| `scripts/diagnostics/e2e_step3_selftest.py` | 18/18 | `1fd4ed79…` | not re-run · 18/18 |

Logs: `B3_*.log` and `B3_summary.txt` in `~/tfm_worktrees/e2e-map/data/landing_wfvote_20261125/`. The walk-forward summary JSON is in `kept_r3/`.

**Before the full cycle,** W11 and W12 were run alone (`precheck_w11_w12.py`):
- **try 1** (`precheck_w11_w12_try1.log`, `ed2f73f0…`): 18 of 22 passed. The 4 launch-side W11 gates failed: the check ran before the backgrounded stub `taskset` had written its mark.
  - The marks appeared moments later.
  - No driver process ran and no out-dir appeared; the stub runs nothing.
  - `run_wrapper` now waits up to 20 s for the mark.
- **try 2** (`114c5e5d…`): 22 of 22.

**Afterwards:**
- no driver process left;
- the W11 synthetic trees hold no out-dir;
- no `__pycache__` outside `data/` and none under `tests/fixtures/`;
- the only tracked change in the check tree was `tests/test_step1_screen.py`;
- no new Optuna study (`optuna_before_r3.txt` = `optuna_after_r3.txt`);
- the main tree's status shows only the three relay files of ADDENDUM 3 (`LEDGER.md`, the request, the reply), unchanged since;
- the check tree was removed (`git worktree list`: 0 matches).

### A4.3 Variant B landing list, updated

Same 33 paths as §6 rows 1-33, plus this note. **Nine files changed in this round:**

| # | Path | round 2 | **now** |
|---|---|---|---|
| 21 | `docs/drafts/loop_v1/runH_commands/00_base.sh` | `c301eee0e7cc…` | **`c86dc6f456db…`** |
| 22 | `docs/drafts/loop_v1/runH_commands/01_sieve_on.sh` | `8806ddd09006…` | **`e993ab71a714…`** |
| 23 | `docs/drafts/loop_v1/runH_commands/02_target_hits.sh` | `bed6d7ed007f…` | **`7624b442cf4e…`** |
| 24 | `docs/drafts/loop_v1/runH_commands/03_rankers.sh` | `31987acbd4e8…` | **`5507742348c1…`** |
| 25 | `scripts/diagnostics/e2e_pipeline_caps.py` | `4e89f2f333a2…` | **`6aa27e54a9b1…`** |
| 27 | `scripts/diagnostics/e2e_step3_harness.py` | `d6390a012404…` | **`adde95e63106…`** |
| 31 | `scripts/diagnostics/e2e_step5_shim.py` | `7528880579a6…` | **`302dcec95935…`** |
| 32 | `scripts/diagnostics/e2e_walk_forward.py` | `06595223e2d3…` | **`2d82b63af39c…`** |
| 33 | `scripts/diagnostics/e2e_walk_forward_selftest.py` | `ca844ed44f05…` | **`a6e2a7c42df0…`** |

- The other 24 rows are as in §0.3 / §6. The vote harness is unchanged (`ed39d3b0…`).
- There is still no force-add.
- e2e-map `git diff --stat` (tracked files): 8 files, +574 −97, plus the untracked new `e2e_pipeline_caps.py`.
- The Beta request X-E2E-WF-VOTE R1 named the round-2 digests. These nine files now differ from what Beta read.

### A4.4 Prior record, corrected (Beta X-E2E-WF-VOTE R1 Finding 1)

My earlier searches covered `docs/` only. The task logs that govern this landing live in `~/distributed_prng_analysis/logs/e2e_session/`, which I did not search. Now added:

```
/bin/grep -l -i 'walk.forward\|e2e_walk_forward\|vote harness\|e2e_vote_harness\|runH_commands\|LAND_WF_VOTE' \
  ~/distributed_prng_analysis/logs/e2e_session/*.txt
```

Documents in that folder that decide or record the subject, each read at the hit:
- `TASK_LAND_WF_VOTE_20261125.txt` and `_ADDENDUM_1.txt` … `_ADDENDUM_4.txt`: this task and its owner and lead requirements.
- `LEAD_ANSWERS_WF_20261123.txt`:
  - the driver was accepted as a draft; no run without the owner's go;
  - Q2: run order smoke → stride 4 → rest;
  - Q5: `prev_block_rank` is the default;
  - MEANWHILE: the three variant command lines (per-seed sieve on, `holdout_hits`, rankers with negatives 200).
  - The wrappers implement these.
- `TASK_LOOP_COMMON.txt`: the walk-forward frame (a lead decision and a setting), no landing by sessions, and that several worktrees edit `meta_prediction_optimizer_anti_overfit.py`.
- `OWNER_GO_wf_smoke_20261124.txt:3-4`: the owner's go for the one-block smoke, and his confirmation that the 2026-11-23 10:40 ruling covers `TB_RULING_S179_IMPLEMENTATION_AUTH.md:546`.
- `OWNER_RULINGS_pending_list_20261124.txt:25`: **"after the first block, the next walk-forward run is the comparison with and without per-seed sieve evidence (variant 01_sieve_on against 00_base)"**.
  - **Consequence of landing variant B:** a tree built from main has no LL-2, so `01_sieve_on.sh` refuses there, by design (A4.1).
  - The owner's next ruled run needs either a tree with the LL-2 files (e2e-map today) or variant A's pipeline files landing first.
  - **This is the owner's call, and I flag it here.**
- `TASK_GRADER_DUP_DRAW_20261124.txt`, `TASK_GRADER_LANDING_READY_20261124.txt`: the grader landing these files repin to.

Labels: all of the above are ALREADY RECORDED. The `OWNER_RULINGS:25` consequence is a status of variant B, not a new ruling. CHANGES A RULING: none.

### A4.5 Limits of this round

- **`--check-only` and the launch still differ.** A wrapper exit of 0 with `started` means the preflight passed and the run was launched under nohup. The run's own outcome is in its log. A refusal that can only be found while running (for example a GPU guard stop) still happens in the background, as before.
- **`meta_prediction_optimizer_anti_overfit.py` on main keeps argparse's default `allow_abbrev`.** It is a pipeline file, and variant B lands none. Only the driver's preflight and the shim's survivors guard stand in front of it. A direct call of the module is outside this driver.
- **The W11 non-vacuity side and W10/G19 read `~/tfm_worktrees/e2e-map`** (as before). W11 also reads the run H inputs under `~/skipsw_run_thr3030`, read only, and fails its precondition gate if they are missing.

---

## 0. ADDENDUM 2 (lead, 07:50): the vote harness's own LL-1 check; variant B re-proven

Task: `~/distributed_prng_analysis/logs/e2e_session/TASK_LAND_WF_VOTE_20261125_ADDENDUM_2.txt`. It closes §7 point 2 below. Variant A was not touched.

### 0.1 What changed

**1. `scripts/diagnostics/e2e_vote_harness.py`**
- New `ll1_check(config_path, root=None)` (`:561`), called in `main` (`:1155`).
  - It runs right after the argument checks, before `prediction_generator` is imported, before any input is read, and long before the out-dir is made (`:1279` region).
  - It reads the `--config` request through `e2e_pipeline_caps.requested_ll1`.
  - It checks LL-1 against the `prediction_generator.py` and `models/feature_manifest.py` of `REPO_ROOT` (`:131`), which is the repo the harness then imports from.
  - A missing switch raises `HarnessError`, which is exit 2 at `:1398`: `[H6] FAIL-CLOSED: pipeline switch missing, nothing run: LL-1 switch needed by {'schema_check': ...} is missing: <file>: ...`.
- If `e2e_pipeline_caps.py` is absent beside the harness, a non-off request refuses ("cannot be checked") and an off config proceeds, with that recorded.
- After `build_config` (`:1212`), the record stores `applied_schema_check` = the `schema_check` the `PredictionConfig` actually carries. If a request does not match it, the run refuses.
- The run record gains `pipeline_switches: {"LL-1": {requested, pipeline_has_switch, files_checked, applied, applied_schema_check}}` (`:1345`). It is outside `config_identity_inputs`, so `config_id` and the pools are unchanged.
- The module is loaded with `dont_write_bytecode` set.

**2. `tests/test_e2e_vote_harness.py`: gate G19 (`g19_ll1_switch`, `:970`) and mutant M25 (`:1290`).** G18 and M19 were already used in sibling tests, so I took G19/M25.
- **Refusal:** on this repo's `HEAD` `prediction_generator.py` (+ manifest if committed), `ll1_check` refuses and names "LL-1" and the file.
- **Non-vacuity:** the same request is accepted on files that have LL-1. These come from `E2E_VOTE_CAPS_WITH_ROOT`, default `~/tfm_worktrees/e2e-map`.
- **End to end:**
  - if the harness's own repo lacks LL-1, `main` with that config raises `HarnessError` and no out-dir exists;
  - if it has LL-1, the run succeeds and records `applied_schema_check: fail`.
- **Default config:** the record says `requested {}` and `applied_schema_check off`.
- **M25** replaces `ll1_check` with a pass-through. It is killed and credited by G19 (hits 1).

**3. `dont_write_bytecode`** is now set in `test_e2e_vote_ll5.load_rev4` and `test_e2e_vote_ordering.load_copy`.

**4. Knock-on edits, each required and each kept tight:**
- **ll5 G15a and ordering G16a** compared the record key sets against the frozen rev 4 and rev 5 runs, and failed on the new key (first try, `B2_try2/`: `record keys differ: {'pipeline_switches'}`).
  - They now require the key sets to differ by exactly `{'pipeline_switches'}`, and that key to show the default (`requested {}`, `applied_schema_check off`).
  - Pools, diagnostics and identity inputs are still compared byte for byte, and are unchanged.
- **`e2e_walk_forward_selftest.snapshot_vote`** (`:121`) now also copies `e2e_pipeline_caps.py` into the vote snapshot, so the pinned snapshot harness runs the check too.

**First tries (kept, not credited):**
- `B2_test_e2e_vote_harness_try1.log`: M25 (then M19) was uncredited with `FileExistsError`, because the gate's scratch dir was reused on its second call; it is now one dir per call;
- `B2_try2/`: G15a and G16a failed as above.

### 0.2 Variant B re-proof on a fresh detached copy of main

- **Setup:** `git worktree add --detach ~/tfm_worktrees/_landing_check_wfvote main` (= `fe5ac3b`). Only the 33 list-B files were copied in (`list_B_r2.txt`, same paths as `list_B.txt` `16413bf8…`; final copy record `copied_B_r2.txt` `f1898df2…`).
- **How the suites ran:** one at a time under `taskset -c 0-5 nice -n 10`, with `python3 -u -B`, `PYTHONDONTWRITEBYTECODE=1` and `CUDA_VISIBLE_DEVICES=""`.

| Suite | Result now | log sha256 | Before (round 1, history) |
|---|---|---|---|
| `tests/test_e2e_vote_harness.py` | **gates 15/15, mutants 17/17** (incl. G19, M25) | `9c76190a…` | 14/14, 16/16 (`2f461496…`) |
| `tests/test_e2e_vote_ll5.py` | 3/3, 3/3 | `a26fed48…` | 3/3, 3/3 (`e9fb8c42…`) |
| `tests/test_e2e_vote_ordering.py` | 6/6, 3/3 | `d3a54ae0…` | 6/6, 3/3 (`240e9856…`) |
| `tests/test_e2e_miss_report.py` | 7/7, 5/5 | `3a617e39…` | 7/7, 5/5 (`28ac7f06…`) |
| `tests/test_step1_screen.py` | GATES 15/15, MUTANTS 13/13, SENTINEL PASS | `f625eccb…` | same (`028265ee…`) |
| `scripts/diagnostics/e2e_walk_forward_selftest.py` | **77/77**, 735 s | `bbc5d111…` | 77/77 (`29d320cf…`) |

Logs: `B2_*.log`, `B2wf_*.log` and the summaries `B2_summary.txt`, `B2wf_summary.txt`, in `~/tfm_worktrees/e2e-map/data/landing_wfvote_20261125/`.

**Afterwards:**
- no `__pycache__` outside `data/`, and none under `tests/fixtures/`;
- the only tracked change was `tests/test_step1_screen.py` (in the list);
- no new Optuna study (`optuna_before_r2.txt` = `optuna_after_r2.txt`, 125 entries);
- the main tree's status is empty;
- the check tree was removed (`git worktree remove --force`; `git worktree list` shows no `_landing_check`).

**Not verified:** I meant to read a vote-run record from inside the walk-forward selftest to see the snapshot harness's `pipeline_switches`. The record I opened was a driver `run_record.json`, and the tree was removed before I noticed. The harness's own record is gated in G19 (default and LL-1 cases); the snapshot copying the module is not separately gated [UNVERIFIED].

### 0.3 Variant B landing list, updated

Same 33 paths as §6 rows 1-33, plus this note. These five sha256 changed (old → new):

| # | Path | old (round 1) | **new** |
|---|---|---|---|
| 5 | `scripts/diagnostics/e2e_vote_harness.py` | `259574e48aee…` | **`ed39d3b00b59…`** |
| 8 | `tests/test_e2e_vote_harness.py` | `44127f4c192c…` | **`d8a0074c2e2e…`** |
| 9 | `tests/test_e2e_vote_ll5.py` | `85564a879bd8…` | **`ab1fd942e439…`** |
| 10 | `tests/test_e2e_vote_ordering.py` | `be9d47cc5df3…` | **`aa8a8f4f5a99…`** |
| 33 | `scripts/diagnostics/e2e_walk_forward_selftest.py` | `d863d4881829…` | **`ca844ed44f05…`** |

- All other rows are unchanged.
- There is still no force-add for B.
- `git diff --stat` in the worktrees:
  - e2e-vote: 4 files, +155 −15, plus new fixtures and the screen copy;
  - e2e-map: 7 files, +168 −28, plus `e2e_pipeline_caps.py`.
- The vote harness sha256 is now `ed39d3b0…`. It was `bbb04e81…` on the branch and `259574e4…` after round 1. The run H scripts compute it live (`V=`).
- **Variant A was not re-run.** Its rows 1-33 would now be these files, but its §5 results are from round 1.

---

## 1. Pipeline switches under variant B (owner ADDENDUM_1, stated first)

### 1.1 What happened on a fresh main copy before the fix (measured, not inferred)

- **Setup:** a detached worktree of `fe5ac3b` with the variant B files copied in. The driver ran on the walk-forward selftest's own fixture: one origin, `lightgbm`, CPU, under `taskset -c 0-5 nice -n 10`.
- **Why the fixture:** the run H scripts need `--step3-device cuda`, run H data and hours. Each variant was therefore run as the driver option that makes it different: `01` = `--step3-extra-args "--per-seed-sieve-features on"`, `02` = `--target-field holdout_hits`, `03` = `--ranker-arms ...`.
- **Record:** `probe_today_B.json`, `probe_today_B.log`. Probe script: `probe_ll.py`.

**Result: every case, base included, did the same thing.**
- **Exit code:** 3.
- **Message:** `[DEVELOPMENT] STEP FAILED (stopped, no retry): o1/step3: exit 2`. The Step 3 log says `REFUSED: signature of full_scoring_worker.score_survivors changed: [... 'holdout_history'] != [... 'holdout_history', 'per_seed_sieve_features', 'fallback_metadata_merge']`.
- **How far it got:** through `run_record.json`, `PLAN.md`, origin 1 `copy/`, `split/`, `_env/` and the Step 3 log, 17 files in all. It stopped at origin 1, Step 3.
- **Cause:** `e2e_step3_harness.py` required the LL-2 parameters in `score_survivors` unconditionally (old `:103-106`) and always passed them (old `:504-505`).

So under plain variant B:
- **00 base did not run.** It refused late, with a message that names a signature, not a switch.
- **01, 02, 03 and every LL option failed the same way, after writing.**
- Nothing ran silently under the wrong label, but nothing refused early either.

Options that map to an LL switch:
- **LL-2:** `--step3-extra-args` with `--per-seed-sieve-features on` (variant 01) or `--fallback-metadata-merge shared`.
- **LL-3:** `--step5-extra-args` or `--nn-args` with `--exclude-features`, `--include-feature-groups` or `--exclude-feature-groups`. No variant script uses these.
- **LL-1:** a `--vote-config` whose prediction section sets `schema_check` to anything but `off`. No variant script uses it.
  - Main's `PredictionConfig.from_json` reads named keys only (`prediction_generator.py:142-…` at `fe5ac3b`), so **on main that key is silently ignored**. Before this fix, a vote with `schema_check` would have run unchecked.
- **02** (`--target-field`, implemented in the shim) and **03** (rankers, which use `load_survivors_with_features`, present on main at `meta_prediction_optimizer_anti_overfit.py:610`) **need no switch.**

### 1.2 The fix

**New file `scripts/diagnostics/e2e_pipeline_caps.py`.**
- It reads the pipeline files the caller is about to use as source (Python `ast`; nothing imported or executed). It consults no version string and no worktree name.
- `capabilities()` (`:59`) checks for each switch:
  - **LL-1:** `prediction_generator.py` has the `PredictionConfig.schema_check` field and `from_json` reads `'schema_check'`; `models/feature_manifest.py` defines `check_rows_against_schema`.
  - **LL-2:** `full_scoring_worker.py` `score_survivors` takes `per_seed_sieve_features` and `fallback_metadata_merge`; `survivor_scorer.py` defines `metadata_merge_fields`.
  - **LL-3:** `meta_prediction_optimizer_anti_overfit.py` declares the three flags; `feature_manifest.py` defines `resolve_step5_exclusions`.
- `requested_ll1/2/3()` (`:135-160`) pick the non-default requests out of the arguments and the vote config. A request equal to the default is not a switch.
- `check()` (`:161`) raises `MissingSwitch`, naming the switch, the request and each file with what it lacks.

**Driver `e2e_walk_forward.py`:**
- `pipeline_switches()` (`:533`) runs inside `build_declaration` (`:562`), immediately after the pin checks. Every run, `--plan-only` included, passes through it **before the out-dir is created**.
- The files checked are the ones actually called:
  - LL-1: the vote harness's repo (it imports its own `prediction_generator`);
  - LL-2 and LL-3: the driver's own repo.
- A refusal is `[DEVELOPMENT] REFUSED: pipeline switch missing, nothing run: ...`, exit 2.
- The declaration records `pipeline_switches` (`:709`): requested, `pipeline_has_switch`, files checked, and applied. It is in `run_record.json`, so every output's label comes from the settings applied.

**Step 3 harness `e2e_step3_harness.py`:**
- `check_signatures` (`:188`) accepts main's signature or main's plus `LL2_SIGNATURE_TAIL` (`:184`), still exact and fail-closed.
- A non-default LL-2 request on a pipeline without the switch refuses before the harness's out-dir exists (`:619-630`), exit 2.
- The two LL-2 arguments are passed only when the pipeline has them (`:518`). Otherwise nothing is passed, which is the default `off`/`legacy6`.
- The record gains `applied_parameters` (`:782`): whether the switch is present, what was passed, and the effective values.

### 1.3 What happens on a fresh main copy after the fix (measured; `probe_fixed_B.json`, `probe_fixed_B.log`)

| Case | Exit | Time | Out-dir | Message / outcome |
|---|---|---|---|---|
| 00 base | **0** | 22.3 s | written (83 files) | `done: 1 origins, 2 table rows`; record: LL-1/2/3 `requested {}`, `pipeline_has_switch false`, applied "default"; Step 3 `applied_parameters.passed_to_score_survivors {}` |
| 01 sieve on (`--per-seed-sieve-features on`) | **2** | 0.4 s | **none** | `REFUSED: pipeline switch missing, nothing run: LL-2 switch needed by {'--per-seed-sieve-features': 'on'} is missing: <tree>/full_scoring_worker.py: score_survivors has no parameter per_seed_sieve_features; ... fallback_metadata_merge; <tree>/survivor_scorer.py: no metadata_merge_fields` |
| LL-2 `--fallback-metadata-merge shared` | **2** | 0.4 s | none | same, naming `{'--fallback-metadata-merge': 'shared'}` |
| 02 target hits | **0** | 21.4 s | written | done (no switch needed) |
| 03 rankers | **0** | 30.6 s | written | done, 3 table rows (no switch needed) |
| LL-3 `--exclude-features ...` | **2** | 0.4 s | none | `LL-3 switch needed by {'--exclude-features': ...} is missing: <tree>/meta_prediction_optimizer_anti_overfit.py: no --exclude-features option; ...; <tree>/models/feature_manifest.py: no resolve_step5_exclusions` |
| LL-3 `--exclude-feature-groups dead` | **2** | 0.4 s | none | same pattern |
| LL-3 `--include-feature-groups ...` | **2** | 0.4 s | none | same pattern |
| LL-1 vote config `schema_check: fail` | **2** | 0.4 s | none | `LL-1 switch needed by {'schema_check': 'fail'} is missing: <vote repo>/prediction_generator.py: PredictionConfig has no schema_check field; ... from_json does not read 'schema_check'; <vote repo>/models/feature_manifest.py: no check_rows_against_schema` |

In summary:
- **Under variant B, `00`, `02` and `03` run on main's unmodified pipeline files.**
- **`01` and every LL option refuse in 0.4 s with exit 2, name the switch and the file, and write nothing.**
- I ran each variant's distinguishing driver option on the fixture, not the `runH_commands/0*.sh` scripts themselves. Those need a card and run H.
  - The scripts reach the same check, because they call the same driver.
  - `01_sieve_on.sh` therefore refuses under B, and `00`, `02` and `03` do not.

### 1.4 Selftest cases added: W10 in `e2e_walk_forward_selftest.py` (`w10_switches`, `:246`)

**For each of LL-1, LL-2 and LL-3:**
- **(a) precondition:** the "without" files (git show of `--caps-without-rev`, default `HEAD`) lack the switch, and the "with" files (`--caps-with-root`, default `~/tfm_worktrees/e2e-map`, as the owner asked) have it.
- **(b) refusal:** `check()` refuses on the "without" files, naming the switch and the file.
- **(c) non-vacuity:** the same request passes on the e2e-map worktree's files and is recorded as applied.
- **(d) end to end on the tree it runs in:**
  - if that tree lacks the switch, the driver refuses with exit 2, no out-dir, and the switch named;
  - if it has the switch, the driver `--plan-only` exits 0 and `run_record.pipeline_switches[sw].applied` equals the request.

**Plus one more case:** the Step 3 harness run directly with `--per-seed-sieve-features on` refuses before its out-dir exists, when the tree lacks LL-2.

**Results:**
- **Variant B:** all 13 W10 gates pass (`B_e2e_walk_forward_selftest.log:103-115`). The selftest is 77 of 77; it was 64 gates before.
- **Variant A:** all 12 applicable W10 gates pass (`A_e2e_walk_forward_selftest.log:103-114`); the direct-harness case prints INFO as not applicable. The selftest is 76 of 76.

---

## 2. Plain summary

- **Repin (task Step 1): done.**
  - The grader pin `c7fe48bf…` is replaced by `3ab8ce2a5330fe36ae638e8bf2915e23bdce08bcefc0221eb47e21177fc8f341`.
  - Every `~/tfm_worktrees/e2e-grader/...` and `~/tfm_worktrees/e2e-vote/...` read now resolves from the file's own location.
  - Sites:
    - driver `:470`, `:477`;
    - selftest `:64-65`;
    - the four `runH_commands/0*.sh`: lines 17-18 `REPO=$(... BASH_SOURCE ...)`, line 21 `V=`, line 22 `G=`;
    - `tests/test_e2e_vote_ordering.py:51-52`, `:68`, `:12`;
    - `tests/test_e2e_vote_harness.py:56`;
    - comments in `e2e_vote_harness.py:93`, `:120`;
    - a prepared copy of main's `tests/test_step1_screen.py` with `:55` → `os.path.join(ROOT, "scripts", "diagnostics", "e2e_vote_harness.py")`. It sits in the e2e-vote worktree as `tests/test_step1_screen.py`; its only difference from `fe5ac3b` is line 55.
  - The grep afterwards found no read of an old path or pin. The remaining hits are prose status lines ("worktree e2e-vote", `e2e_vote_harness.py:5-6`, `e2e_miss_report.py:5`) and `scripts/owner_land.py`, which is the landing tool itself.
  - Dated READY notes keep the old pin as history and were not edited.
- **The vote harness's sha256 changes.** The two comment lines move it from `bbb04e81…` to `259574e48aee…`. Nothing pins `bbb04e81` in code: the driver takes `--vote-harness-sha256` from the scripts, and they compute it live (`V=`). Earlier READY notes cite `bbb04e81` as history.
- **Fresh-main proof (task Step 4): both variants pass every suite.**
  - Variant B: 13 suites.
  - Variant A: 13 suites plus the LL suite.
  - Variant A's default output equals main's, and the non-vacuity check shows the comparison detects a real switch (§5).
- **Recommendation, following the owner's choice: land variant B (33 files, no force-add).** The six pipeline/manifest files stay on `ch14-learn` until they land on their own.

## 3. Hidden files (task Step 2)

**(a) Ignored files outside `data/` that hold work**

| Branch | Path | What | Action |
|---|---|---|---|
| e2e-map | `models/feature_manifest.py`, `models/feature_manifest.json` | tracked on the branch (WIP snapshot), but `.gitignore:162 models/` (and `:41 *.json`) match | **force-add, variant A only** |
| e2e-map | `logs/` (368 K): `e2e_runH_wf_0{0,1,2}_*.log/.cmd/.pid`, `e2e_runH_step3_*.log`, `e2e_runH_build.log`, `logs/e2e_wf/` | run logs cited by `READY_WF_SMOKE/SIEVE_ON/TARGET_HITS` notes | keep as evidence; not code |
| e2e-vote | `logs/e2e_loop_miss/`, `logs/e2e_vote/`, `logs/e2e_vote_npz/` (1.1 M) | run logs cited by `READY_H6`, `READY_LOOP_MISS`, `NOTE_VOTE_NPZ` | same |
| both | `__pycache__/`, `utils/__pycache__/`; e2e-vote also `scripts/diagnostics/__pycache__/` (`e2e_vote_harness`, `e2e_miss_report`), `tests/__pycache__/`, `integration/`, `models/`, `models/wrappers/` | stray bytecode from earlier runs | no work; not landed |
| ch14-learn (outside this landing) | `models/inference_bundle.py` | **ignored and untracked**; needed by ch14-learn's `prediction_generator.py` (LL-6) | not in this landing, but at risk of being lost: it is not in any commit |

**(b) Tests that read untracked `data/` or a `daily3*.json`**

| Test | Reads | Status |
|---|---|---|
| `tests/test_e2e_vote_ll5.py:51` | `data/e2e_loop_miss/provenance/e2e_vote_harness_rev4.py.txt` (sha `99791061…`) | **moved** to `tests/fixtures/e2e_vote_harness_rev4.py.txt`, same bytes; test repointed |
| `tests/test_e2e_vote_ordering.py:49` | `.../e2e_vote_harness_rev5.py.txt` (sha `a26e39c1…`) | **moved** to `tests/fixtures/e2e_vote_harness_rev5.py.txt`; repointed |
| `tests/test_ll_feature_settings.py:34-39` (ch14-learn, variant A) | `~/tfm_worktrees/e2e-map/data/e2e_runH/{step3_full,step45/winner,copy}`, `~/skipsw_run_thr3030/{daily3.json,optimal_window_config.json}`, read only, by absolute path | **not changed** (ch14-learn's file). It passes only while those exist; after landing it still reads another worktree's untracked data |
| `scripts/diagnostics/e2e_step3_selftest.py:62-71` | `~/skipsw_run_thr3030/{scoring_chunks,...}`, stat only (no-write watch list) | fine if absent |
| `e2e_walk_forward_selftest.py` W10 | `--caps-with-root` default `~/tfm_worktrees/e2e-map` (tracked files there, not `data/`) | owner-directed; fails clearly if that worktree is gone |
| `runH_commands/0*.sh:27` | `data/e2e_runH/copy/bidirectional_survivors_binary.npz` relative to the repo, plus `~/skipsw_run_thr3030` | e2e-map's untracked run copy. From any other tree these scripts fail at the npz. Not a test, but a hidden input |

All other suites build their fixtures under `data/` themselves.

## 4. The six pipeline/manifest files (task Step 3)

**(a) Byte compare, e2e-map against ch14-learn** (worktree = HEAD in both):

| File | e2e-map | ch14-learn | main `fe5ac3b` |
|---|---|---|---|
| `full_scoring_worker.py` | `3f8d3d05…` | **same** | `2c1c9639…` |
| `meta_prediction_optimizer_anti_overfit.py` | `16cd3ae5…` | **same** | `dfc53e67…` |
| `prediction_generator.py` | `3cc11eb1…` | **`4f6a26bf…` differs** | `53e9fe71…` |
| `survivor_scorer.py` | `efc08ae3…` | **same** | `a53a873f…` |
| `models/feature_manifest.py` | `a454e311…` | **same** | absent |
| `models/feature_manifest.json` | `b2fabc87…` | **same** | absent |

- **The `prediction_generator.py` difference is LL-6 on ch14-learn.** The body of `_load_nn_scaler` (e2e-map `:397-454`, the CAT-B checkpoint read and the clamping) is replaced by a call to `models.inference_bundle.load_nn_scaler`, which sets the same three attributes. Nothing else differs.
- That module is the ignored, untracked file in §3(a).
- Variant A here uses e2e-map's copy, which is LL-1 to LL-3 only.
- Main changed none of these files between the merge base `41ad509` and `fe5ac3b` (`git diff --name-only 41ad509 fe5ac3b`).

**(b) What needs them:**
- **Before the fix, every driver run did, base included** (§1.1).
- **After the fix, only these need them:**
  - LL-2 in `--step3-extra-args` (variant `01`);
  - LL-3 in `--step5-extra-args` or `--nn-args`;
  - LL-1 in the vote config;
  - W10's "present" side.
- **Base mode, `02` and `03` run on main's unmodified pipeline files** (§1.3).
- Step 3, 5 and 6 behaviour on main's files is exercised by the walk-forward selftest, the step3 selftest and the vote harness suites, all of which pass under B.

## 5. Results (task Step 4)

- **Setup:** `git worktree add --detach ~/tfm_worktrees/_landing_check_wfvote main` (= `fe5ac3b`). Only the landing-list files were copied in (`copy_in.sh`; lists `list_B.txt` `16413bf8…` and `list_A.txt` `da1580bb…`; copies `copied_B.txt`, `copied_A.txt`).
- **How the suites ran:** one at a time (`run_suites.sh`), under `taskset -c 0-5 nice -n 10`, with `python3 -u -B`, `PYTHONDONTWRITEBYTECODE=1`, venv `~/venvs/torch` and `CUDA_VISIBLE_DEVICES=""`.
- **Variant A** is the same tree after variant B, with the 8 extra files copied over it.

| Suite | Variant B | log sha256 | Variant A | log sha256 |
|---|---|---|---|---|
| `tests/test_e2e_vote_harness.py` | gates 14/14, mutants 16/16 | `2f461496…` | 14/14, 16/16 | `44499a03…` |
| `tests/test_e2e_vote_ll5.py` | 3/3, 3/3 | `e9fb8c42…` | 3/3, 3/3 | `e7ddd994…` |
| `tests/test_e2e_vote_ordering.py` | 6/6, 3/3 | `240e9856…` | 6/6, 3/3 | `771fb79d…` |
| `tests/test_e2e_miss_report.py` | 7/7, 5/5 | `28ac7f06…` | 7/7, 5/5 | `d2a0a84f…` |
| `tests/test_step1_screen.py` (repinned copy) | GATES 15/15, MUTANTS 13/13, SENTINEL PASS | `028265ee…` | same | `f4b6d186…` |
| `tests/test_e2e_grader.py` | 37/37 | `63edd0e5…` | 37/37 | `45645059…` |
| `tests/test_compare_settings.py` | 9/9 | `9f88e371…` | 9/9 | `c4f4e10a…` |
| `tests/test_switch_check.py` | 12/12 | `86168dc8…` | 12/12 | `8e6a247c…` |
| `tests/test_registry_repair.py` | 5/5 | `442017d5…` | 5/5 | `442017d5…` |
| `tests/mutate_e2e_grader.py` | 89/89 killed and credited | `eee11a61…` | 89/89 | `24238952…` |
| `tests/test_window_anchor_brief2.py` | GATES 18/18, MUTANTS 12/12, PASS | `4ede8f91…` | same | `dff81686…` |
| `scripts/diagnostics/e2e_walk_forward_selftest.py` | **77/77** (incl. 13 W10), 702 s | `29d320cf…` | **76/76** (12 W10), 716 s | `93203592…` |
| `scripts/diagnostics/e2e_step3_selftest.py` | 18/18 | `521bf2e3…` | 18/18 | `37d71377…` |
| `tests/test_ll_feature_settings.py` | n/a | | **35/35, mutants 10/10**; `LL2.default_bit_identical_to_HEAD` `b00c34c1…` both, HEAD = `fe5ac3b` | `83996e83…` |

Logs: `B_*.log` and `A_*.log`, with summaries in `B_summary.txt`, `A_summary.txt` and `A_ll_summary.txt`.

**Existing suites for Steps 3, 5 and 6 on main.** The only main tests that import the four pipeline modules are:
- `tests/test_step1_screen.py`, which is run above;
- `tests/phase6/wall_ab_gate.py`, which needs GPUs and rigs and was not run.

The Step 3, 5 and 6 coverage under A therefore comes from the LL suite (Step 3 bit-identity against `fe5ac3b`'s own files, Step 5 LL-3, Step 6 LL-1), the step3 and walk-forward selftests, and the vote harness suites.

**Default output under A against main (non-vacuity per lesson 3)** (`compare_default.py`; `compare_A_vs_main.json`, `compare_main_repeat.json`):
- **Runs compared:** one-origin base driver runs on the fixture,
  - on main's pipeline (B tree);
  - repeated on main's pipeline (the control);
  - on the LL pipeline (A tree).
- **Fingerprint:** the selftest's own (`fingerprint()`), covering 11 keys: Step 3 rows minus timestamps, the label and train blocks, the Step 5 winner, the cv ranking, the checkpoint sha, the vote history and seal, and the pools per arm.
- **A base against main base: 0 of 11 keys differ.** The per-draw outcomes are equal.
  - The 72 per-draw grade rows are equal apart from `config_id`.
  - `config_id` also differs between main and its own repeat (it carries the run), so it is not a pipeline effect.
- **Non-vacuity:** on the same A tree, `--per-seed-sieve-features on` against base gives **5 of 11 keys different**: `step3_survivors`, `cv/s42/lightgbm`, `checkpoint/s42/lightgbm`, `winner/s42` and `pools/A1s42_o1_p20`. The comparison detects a real switch.
- **A probe of all 9 cases on the A tree** (`probe_A.json`):
  - 00, 01, LL-2 shared, 02, 03, LL-3 exclude-features and exclude-groups: exit 0.
  - LL-3 `--include-feature-groups sieve`: exit 3. **The group name was my probe's error** (the manifest calls it `per_seed_sieve`); the manifest refused it as `unknown feature group`.
  - LL-1 `schema_check: fail`: exit 3. See §7.

**Afterwards:**
- the check tree had no `__pycache__` outside `data/`, and the only differences from `fe5ac3b` were the landing files;
- `~/distributed_prng_analysis/optuna_studies/` has the same 125 entries before and after (`optuna_before.txt`, `optuna_after.txt`), and `find -newer B_start.stamp` is empty;
- the main tree's `git status --porcelain` is empty;
- `git worktree remove --force ~/tfm_worktrees/_landing_check_wfvote` succeeded, and `git worktree list` no longer shows it;
- frozen read-only model dirs needed `chmod -R u+w` first;
- the selftest summaries, probe logs and run records were copied to `landing_wfvote_20261125/kept/` before removal.

## 6. Landing lists (repo-relative, against main `fe5ac3b`)

**Variant B = rows 1-33. Variant A = rows 1-41.**
- `src` is the worktree to take the file from: vote = `e2e-vote`, map = `e2e-map`, learn = `ch14-learn`.
- `N` = new on main, `M` = modifies a main file.

| # | Path | src | N/M | sha256 |
|---|---|---|---|---|
| 1 | `docs/drafts/e2e_v1/READY_H6_VOTE_HARNESS_20261123.md` | vote | N | `fbc01afc537d…` |
| 2 | `docs/drafts/loop_v1/NOTE_VOTE_NPZ_20261124.md` | vote | N | `7913429cde52…` |
| 3 | `docs/drafts/loop_v1/READY_LOOP_MISS_20261123.md` | vote | N | `07da565698af…` |
| 4 | `scripts/diagnostics/e2e_miss_report.py` | vote | N | `ec7c7ad16b25…` |
| 5 | `scripts/diagnostics/e2e_vote_harness.py` | vote | N | `259574e48aee…` (was `bbb04e81…`; comments only) |
| 6 | `scripts/diagnostics/e2e_vote_npz.py` | vote | N | `4c2f34998077…` |
| 7 | `tests/test_e2e_miss_report.py` | vote | N | `ffc7250b450e…` |
| 8 | `tests/test_e2e_vote_harness.py` | vote | N | `44127f4c192c…` |
| 9 | `tests/test_e2e_vote_ll5.py` | vote | N | `85564a879bd8…` |
| 10 | `tests/test_e2e_vote_ordering.py` | vote | N | `be9d47cc5df3…` |
| 11 | `tests/fixtures/e2e_vote_harness_rev4.py.txt` | vote | N | `99791061cde4…` |
| 12 | `tests/fixtures/e2e_vote_harness_rev5.py.txt` | vote | N | `a26e39c1ab49…` |
| 13 | `tests/test_step1_screen.py` | vote | **M** | `31ae4b3450d2…` (line 55 only) |
| 14 | `docs/drafts/e2e_v1/MAP_E2E_THIN_PATH_20261122.md` | map | N | `e771a726300d…` |
| 15 | `docs/drafts/e2e_v1/READY_H3_STEP3_HARNESS_20261123.md` | map | N | `8e117c5101dc…` |
| 16 | `docs/drafts/loop_v1/READY_LOOP_WF_20261123.md` | map | N | `3ffb04dc0799…` |
| 17 | `docs/drafts/loop_v1/READY_WF_SIEVE_ON_BLOCK0_20261124.md` | map | N | `b4b6082df5cd…` |
| 18 | `docs/drafts/loop_v1/READY_WF_SMOKE_20261124.md` | map | N | `eb8043123e54…` |
| 19 | `docs/drafts/loop_v1/READY_WF_TARGET_HITS_BLOCK0_20261124.md` | map | N | `c2ae30e656c0…` |
| 20 | `docs/drafts/loop_v1/WF_FIRST_TABLE_GUIDE_20261123.md` | map | N | `ee917e55e4b4…` |
| 21 | `docs/drafts/loop_v1/runH_commands/00_base.sh` | map | N | `c301eee0e7cc…` |
| 22 | `docs/drafts/loop_v1/runH_commands/01_sieve_on.sh` | map | N | `8806ddd09006…` |
| 23 | `docs/drafts/loop_v1/runH_commands/02_target_hits.sh` | map | N | `bed6d7ed007f…` |
| 24 | `docs/drafts/loop_v1/runH_commands/03_rankers.sh` | map | N | `31987acbd4e8…` |
| 25 | `scripts/diagnostics/e2e_pipeline_caps.py` | map | N | `4e89f2f333a2…` (new this task) |
| 26 | `scripts/diagnostics/e2e_build_run_copy.py` | map | N | `bd4d1ae8b133…` |
| 27 | `scripts/diagnostics/e2e_step3_harness.py` | map | N | `d6390a012404…` |
| 28 | `scripts/diagnostics/e2e_step3_selftest.py` | map | N | `588bc5496340…` |
| 29 | `scripts/diagnostics/e2e_step5_check.py` | map | N | `5087de318657…` |
| 30 | `scripts/diagnostics/e2e_step5_ranker.py` | map | N | `57e5a3eae364…` |
| 31 | `scripts/diagnostics/e2e_step5_shim.py` | map | N | `7528880579a6…` |
| 32 | `scripts/diagnostics/e2e_walk_forward.py` | map | N | `06595223e2d3…` |
| 33 | `scripts/diagnostics/e2e_walk_forward_selftest.py` | map | N | `d863d4881829…` |
| 34 | `full_scoring_worker.py` | map | M | `3f8d3d056ddc…` |
| 35 | `meta_prediction_optimizer_anti_overfit.py` | map | M | `16cd3ae5fa64…` |
| 36 | `prediction_generator.py` | map | M | `3cc11eb1b3ad…` |
| 37 | `survivor_scorer.py` | map | M | `efc08ae3fd6d…` |
| 38 | `models/feature_manifest.py` | map | N | `a454e3117ae1…` **force-add** |
| 39 | `models/feature_manifest.json` | map | N | `b2fabc87eb62…` **force-add** |
| 40 | `tests/test_ll_feature_settings.py` | learn | N | `2916b1f82f25…` |
| 41 | `tests/ll_step3_runner.py` | learn | N | `a3556cad5992…` (**needed by #40**, `:84`; the task named only #40) |

Plus this note, `docs/drafts/loop_v1/READY_LAND_WF_VOTE_20261125.md` (N).

**Force-add:**
- `git check-ignore -v --no-index` over all 41 paths on main matches only #38 and #39 (`.gitignore:162 models/`; #39 also `:41 *.json`).
- **Variant B needs no force-add.**
- For A: `git add -f models/feature_manifest.py models/feature_manifest.json`. Afterwards, check that `git ls-files models/feature_manifest.*` shows both (`READY_GRADER_LANDING_20261124.md:111` records how one was silently lost).

**Edits made by this task** (all uncommitted in the two worktrees):
- e2e-map: `git diff --stat` shows 7 files, +165 −28, plus the new #25.
- e2e-vote: 4 files, +11 −11, plus the new #11, #12 and #13.

## 7. Surprises and open points

1. **The step3 harness refused main's own `score_survivors`** (§1.1). Without the fix, variant B would have landed a driver that could not run its base mode on main. Every case would have failed late, at Step 3, after writing.
2. **LL-1 is silent on main without the driver.** Main's `from_json` ignores a `schema_check` key. The driver now refuses it (§1.3), but the vote harness run on its own, outside the driver, still accepted such a config and ran without the check. **Closed by ADDENDUM 2, §0.**
3. **LL-1 `fail` under A refuses the unlearned arm.** The A0 `field_sum` arm has no sidecar, and the check raised `ManifestError: [LL-1 schema-check] sidecar has no feature names; sidecar has no feature_schema_hash` (`probe_A.json`, LL1 case, exit 3 at `o1/vote/A0_o1_p20`). A walk-forward with LL-1 on needs the check limited to learned arms. This is a fact about how LL-1 and the driver combine. It is not exercised by any variant.
4. **The driver cannot run from the main tree.**
   - `FORBIDDEN` (`e2e_walk_forward.py:98`) contains `~/distributed_prng_analysis`, and the out-dir must lie under the driver's own `data/` (`main()`).
   - Run from the main checkout after landing, every run refuses ("forbidden root"). Runs need a detached worktree, as run trees already are.
   - The `runH_commands` scripts `cd` to their own repo, so they must be run from such a worktree, and that worktree needs `data/e2e_runH/copy/` (§3(b)).
5. **The repinned driver and selftest no longer run inside the `e2e-map` worktree itself.** Neither file is there (base `41ad509`): the vote harness lives on `e2e-vote` and the grader landed on main. They run only in a combined tree, such as the check tree or main after landing.
6. **Resume of an old run refuses.** The declaration gained `pipeline_switches`, so `--resume` on a run directory written before this change fails closed: "declaration differs".
7. **Bytecode beside the frozen copies.** `SourceFileLoader.exec_module` on the `.py.txt` copies writes bytecode unless `-B` or `dont_write_bytecode` is set. That is the origin of `e2e-vote/data/e2e_loop_miss/provenance/__pycache__/`. **Fixed by ADDENDUM 2:** both loaders now set `dont_write_bytecode`.
8. **The W10 "with" side defaults to `~/tfm_worktrees/e2e-map`,** as the owner specified. After landing, that default reads another worktree. It is an option (`--caps-with-root`), and the gate fails with a clear precondition message if the path lacks the switches.
9. **`tests/test_ll_feature_settings.py`** (variant A) **reads run H inputs by absolute path from the e2e-map worktree's `data/`** (§3(b)).
10. **Operating errors of mine, recorded:**
    - My first suite runner died at once: `set -u` against the venv's `activate`, `LD_LIBRARY_PATH: unbound variable` (`B_runner_attempt1_died.out`). Nothing had run. It was rerun after the fix.
    - The per-case logs of the first two probes were deleted with the check tree's probe directory. Their exit codes, messages and file counts survive in `probe_today_B.json` / `.log` and `probe_fixed_B.json` / `.log`.

Stopped here; waiting for the lead and the owner. No landing, commit or push has been done.

---

## 8. ADDENDUM 3: Beta courtesy review, X-E2E-WF-VOTE R1 (one round, not binding; the owner decides)

- **Task:** `~/distributed_prng_analysis/logs/e2e_session/TASK_LAND_WF_VOTE_20261125_ADDENDUM_3.txt` (owner, 08:27).
- **Request:** `docs/exchange/to_beta/X-E2E-WF-VOTE_R1_ALPHA_REVIEW_REQUEST_variant_b_landing.md` (main tree, untracked; sha256 `b66205b7…`).
- **Run:** `python3 -u -B scripts/beta_relay.py --request <that file>` from the main tree, under nohup, with stdout to `landing_wfvote_20261125/relay_X-E2E-WF-VOTE_R1.out`. I waited on its PID in the foreground.
- **Before the run:** the only processes with a cwd in the main tree were an interactive shell and a read-only `http.server` serving `~/tfm_worktrees`.
- **Verdict: `REVIEW RECEIVED` (exit 0)**, 415.5 s.
  - Reply: `docs/exchange/from_beta/X-E2E-WF-VOTE_R1_BETA_REVIEW.md` (sha256 `8452caa9…`).
  - Relay log: `logs/beta_relay/X-E2E-WF-VOTE_R1.log`.
  - The relay added one ledger row: `docs/exchange/LEDGER.md`, X-E2E-WF-VOTE R1, 2026-11-25 08:38 PST.
- **Main tree status after the run:** `M docs/exchange/LEDGER.md`, plus the untracked request and reply. Nothing else.
- **The review file is the record.** The table below quotes its positions word for word. Nothing was fixed and no second round was started, as instructed.

| Question | Beta's position (verbatim) | Severity (verbatim) | My view, one line |
|---|---|---|---|
| Q1 — Refusal | `OBJECTION` | **LIVE** — `01_sieve_on.sh` writes a log and timing file and reports success when its child refuses the missing LL-2 switch. | Agree, and the label is right. `01_sieve_on.sh:24-40` backgrounds the driver with `nohup … &`, opens `logs/…log` and `/usr/bin/time -o …time` before the driver's check, then echoes `started`, so it breaks ADDENDUM 1's "exit non-zero, write nothing" for a variant script. My probe bypassed the wrapper. The same review raises a second Q1 point: an abbreviated `--per-seed-sieve-f on` passes the driver's exact-name match and fails late in Step 3. That one is real too, but it fails closed after writing, not under a wrong label. |
| Q2 — Record | `OBJECTION` | **LIVE** — A normal Step 5 passthrough can make a target-hits run train on the original `holdout_quality` input while its record reports `holdout_hits` and its input check reports success. | The mechanism is real: the shim counts only the literal `--survivors` token (`e2e_step5_shim.py:134`), so a trailing `--survivors=PATH` or `--k-folds` in `--step5-extra-args` wins at the optimizer's parser. **I would contest "normal" and so the `LIVE` label.** No variant script or note passes either option (the run H scripts use the default `--step5-extra-args`, and only `01` sets `--step3-extra-args`), so I would call it `THEORETICAL`. Under the protocol the label stands as Beta wrote it until the owner rules, and both positions go to him. |
| Q3 — Additive | `NO OBJECTION` | — | Agree. Beta independently checked all 33 digests, the single existing path (`tests/test_step1_screen.py`, line 55 only), and that no destination is ignored. |

**Also in the review (Beta's own words are in the file):**
- **Finding 1:** my request's prior record left out the governing task files ADDENDUM_1/2/3 and `LEAD_ANSWERS_WF_20261123.txt`. That is correct: my searches covered `docs/` only, and those files sit in `logs/e2e_session/`.
- **Q2:** Beta built a fresh vote snapshot and verified that the snapshot harness records `pipeline_switches.LL-1` (`requested {}`, `applied_schema_check off`, path inside the snapshot). That settles my §0.2 **[UNVERIFIED]** item by Beta's run, not mine.
- **Q1, main checkout:** Beta notes the driver refuses there on the forbidden root before it inspects switches. This is recorded in §7 point 4.

**Status for the owner:**
- **Two `LIVE` objections stand: Q1 and Q2.** Under `RELAY_PROTOCOL.md` §2.4 a `LIVE` objection holds the commit until the owner rules. ADDENDUM 3 makes this review courtesy and non-binding, and the owner decides.
- **I contest the Q2 severity** (`LIVE` → `THEORETICAL`) for the reason in the table. Beta's position is that a passthrough of these options is a normal use. That disagreement is the owner's.
- **Q1 I accept as `LIVE`.**
- Nothing was changed after the review *(at the time; see §A4 at the top for the later fixes).*

Stopped. No fix, no second round, no commit, push or landing. *(Superseded by §A4: the lead's ADDENDUM 4 ordered the fixes.)*
