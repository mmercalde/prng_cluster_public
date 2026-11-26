# X-CH14-DIAG  R1  REVIEW_REQUEST  landing of the Chapter 14 diagnostics waves 1-2 (ch14-diag) onto main

- **Review class:** BEFORE-COMMIT. This changes Step 5 (`meta_prediction_optimizer_anti_overfit.py`), its trial worker
  (`train_single_trial.py`) and the post-Step-5 health check WATCHER reads. Lead task:
  `logs/e2e_session/TASK_LEADS_CH14DIAG_BETA_20261125.txt` (19:10). It is not binding; the owner decides.
- **Base commit:** `579dbb7` (main = origin = public). The package was merged and proved on `7f4505a`.
  `git diff --name-only 7f4505a 579dbb7 -- <the 17 files>` is empty: main's two later commits (CLAUDE.md, the leads
  register docs, `agents/registry/parameter_registry.py`) touch none of them.
- **Rounds:** one. Alpha changes nothing after your reply and reports your positions to the lead word for word.
- **Allowed answers per question:** `NO OBJECTION` · `OBJECTION` (with `LIVE` or `THEORETICAL`) · `SUGGESTION` ·
  `NEEDS EVIDENCE`.
- **Scope:** real-world cases on this system as it is today: VM 101, Step 5 as WATCHER runs it with the manifests as they
  are (no `--diagnostics-level` entry yet), `watcher_policies.json` as it is (no `grading_rules` key, so `legacy`).

**The files under review are NOT on main.** Read each one at `~/tfm_worktrees/ch14diag-land/<path>`. That worktree is a
detached checkout of `7f4505a` with the branch merged in. The sha256 is that file's content now.

## Prior record

**Searches run** (main `7f4505a` and `579dbb7`, 2026-11-25 18:25-19:20 PST):

```
S = docs/PROJECT_FILE_CATALOG.md docs/TB_RULING_* docs/TB_RULING_REQUEST_* docs/PROPOSAL_* docs/CHAPTER_*
    docs/*_AUDIT.md docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/LEADS.md docs/exchange/
git grep -l -E "CH14-DIAG|diagnostics-level|grading_rules" 7f4505a -- $S                 -> no hits
ls docs/exchange/{to,from}_beta | grep -iE "ch14|diag"                                    -> no earlier thread
/bin/grep -rnIiE "manual_seed|non-?determinis|not (reproducible|deterministic)|(nn|neural|torch).{0,60}(seed|determinis)" $S
/bin/grep -lE "ch14-diag|CH14-DIAG|ch14_diag" logs/e2e_session/*.txt                      (logs/ is ignored; read on disk)
git diff --name-only 41ad509 1ba3d6f ; git log 41ad509..7f4505a -- <each file>
```

**Documents found that decide, authorise, schedule or record the subject:**
- **The branch's own record:** `docs/drafts/loop_v1/READY_LOOP_DIAG_20261123.md` (wave 1) and
  `READY_LOOP_DIAG_2_20261123.md` (wave 2), and the plan `docs/drafts/ch14_v1/PLAN_CH14_DIAGNOSTICS.md` (prior record
  `:13-96`). All three land with this package.
- **The survey that classified the branch:** `~/tfm_worktrees/e2e-map/docs/drafts/loop_v1/SURVEY_ITEM5_BRANCHES_20261125.md`
  §1. It records 16 files, all unlanded; only the optimizer changed on main since `41ad509` (`7cac105`, LL-3).
- **The lead's tasks:** `logs/e2e_session/TASK_CH14_DIAG.txt`, `TASK_LOOP_DIAG.txt` (built it);
  `TASK_CH14_DIAG_LAND_PREP_20261125.txt` (prepare the landing); `TASK_LEADS_CH14DIAG_BETA_20261125.txt` (this request).
- **The owner's defaults:** `logs/e2e_session/OWNER_RULING_defaults_20261123.txt` (19:23).
  - Items: fixed rules, round base 1, and `full` by default once the manifest entry exists.
  - The condition: each flips only when its piece is landed, with his yes at the time.
  - Nothing in this package flips them.
- **NN determinism:**
  - ALREADY RECORDED in general: "Step 5 training is not" deterministic
    (`~/tfm_worktrees/ch14-learn/docs/drafts/ch14_v1/PLAN_CH14_LEARNING_LOOP.md:209-210`); the unseeded split
    (`docs/CHAPTER_6_AUDIT.md:140`); GPU non-determinism (`docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md:571`).
  - The CPU measurement at the base revision is NEW (READY note, NN section).

**Labels:**

| Item | Label |
|---|---|
| CD-00 to CD-13, CD-23, TODO-CH14-4: the work and its settings | ALREADY RECORDED (the two READY notes; R-60 to R-76 in `docs/CHAPTER_CORRECTIONS_REPORT_20261111.md` §2.11 for the CD-01 to CD-12 items) |
| The 3-way merge onto main, the re-pin to `7f4505a`, the base-precondition test, 10 added mutants, the two strengthened gates, the new Step 5 defaults suite | NEW |
| The NN's run-to-run difference on CPU at `7f4505a`, measured | NEW (the general fact is ALREADY RECORDED, above) |
| CHANGES A RULING | none. Every default stays at today's behaviour |

**Not searched:** the host outside the repository and the named worktrees, ser8 archives, the private repository,
project files outside the repository.

## Files under review (17)

| # | Lands at | N/M on `579dbb7` | sha256 now | Note |
|---|---|---|---|---|
| 1 | `meta_prediction_optimizer_anti_overfit.py` | **M** | `48052f2021ee055e834852b64d04497caa56b542383779d824c9e3e96dbd0e17` | 3-way merge: main's LL-3 + the branch; **pipeline (Step 5)** |
| 2 | `train_single_trial.py` | **M** | `8db47e737183cdcadd0fa698267b583566595eaa09d4a5c4ea3e111414a793b2` | **pipeline (Step 5 trial worker)** |
| 3 | `training_diagnostics.py` | **M** | `bd3787d7a61ccb4604ca4fa2537503bf42fc4dabc01ceaf3fcc4907de9852541` | report producer |
| 4 | `training_health_check.py` | **M** | `ac8ee08ece13f2b7517977287b530376e20680c5ce5d74dc1f5aa25ccc744ad0` | WATCHER's post-Step-5 health check |
| 5 | `diagnostics_llm_analyzer.py` | **M** | `cb73536dbe0ce6dc11b1b59b6db1bca9c6bfa70858700258d4abfb80ad6801c1` | |
| 6 | `parameter_advisor.py` | **M** | `d9bf73dc82af7c640069956ff97edf8d349f44dceeee1573e9c074895cfd17d2` | |
| 7 | `agents/contexts/advisor_bundle.py` | **M** | `95711636feb165781332629a14a15a02b7c53b765c73629b261a8675dd5eb63b` | not hand-landed |
| 8 | `tests/test_ch14_diag_wave1.py` | N | `5fff62b4ee9eaf87184ae85d2bad7e9e01283b3a6db0e7181d11ea31727aac91` | re-pinned |
| 9 | `tests/test_ch14_diag_wave2.py` | N | `81749ce93235b98dcc3a70563726a299b4b3264c6054d373666da1325bd30cf9` | re-pinned |
| 10 | `tests/ch14_wave1_mutants.py` | N | `390bcf4845e205e39aa5e2934e018b527212ce1fd37f187ab25dde0979459944` | +M22-M24 |
| 11 | `tests/ch14_wave2_mutants.py` | N | `37fdc91fdd87f0d2b9f1616c2d0de5a348927f3a874439a8fc8e255d8376f4b7` | +W30-W36 |
| 12 | `tests/ch14_cpu_patch.py` | N | `5e5fc75666da75791c29ff5100b383ea83a6180505e8524d8f6caa1990e9b504` | |
| 13 | `tests/test_ch14_diag_land_defaults.py` | N | `03e67c3f6c4ec64a2637d3f6b67259458be64f981f4ba7db55834e59fc72607f` | **new: the Step 5 defaults gate** |
| 14 | `docs/drafts/ch14_v1/PLAN_CH14_DIAGNOSTICS.md` | N | `9b74b0aee7a2bf577fdafb42e9bad05f028cbcd6c376c6803d26d9aea9cfd118` | |
| 15 | `docs/drafts/loop_v1/READY_LOOP_DIAG_20261123.md` | N | `4ae219bb32fb5ac8eb4f536c2a074710c1630687a6f89290d94c61c34654f3ac` | |
| 16 | `docs/drafts/loop_v1/READY_LOOP_DIAG_2_20261123.md` | N | `5d13a3f9d70e8f55918f698c1fbeb0f41464c1b7b74f17d472b89751a5cd7a9a` | |
| 17 | `docs/drafts/loop_v1/DRAFT_WATCHER_CD07_20261123.diff` | N | `0cfec893a039784f4dffb5e84047bd7213e52be0d78c5b5df1db64236729c398` | a draft diff **for `agents/watcher_agent.py` (hand-landed); not applied** |

**Checks on the 17 files:**
- none is git-ignored (`git check-ignore --no-index`);
- none is in `GOVERNED_FILES` (`scripts/gate12_parity_gate.py:174-185`);
- none is in `HAND_LANDED_FILES` or `HAND_LANDED_DIRS` (`scripts/owner_land.py:86-87`).

**Also to read:**
- **The READY note:** `~/tfm_worktrees/ch14diag-land/docs/drafts/loop_v1/READY_LAND_CH14_DIAG_20261125.md`, sha256
  `bf91a49f7500766a2ea070db1a707cf757ddc8240b002baa94af240a406fa600`. It lands with the files.
  - "The merge" and "Test re-fit" sections: what changed and why.
  - "Incident": a harness write through a symlink, found by the suite's own gate, undone and prevented.
  - "NN non-reproducibility": the measurement and what the NN gates do and do not claim.
- **The evidence:** `~/tfm_worktrees/ch14diag-land/data/ch14_land/` (ignored `data/`).
  - `logs/`: every suite log, `nn_child_argv_check.txt`, the status snapshots;
  - `defaults_final/summary.json`: the final defaults run;
  - `defaults_all{3,4,5}/`: the intermediate runs;
  - `mutants_w1/`, `mutants_w2/`: summaries and per-mutant logs.

**About running anything:** the tests write under `<worktree>/data/` (the suite's `--out` must be there). A run inside
`ch14diag-land` writes into it; a write into the main repository during your review is a `BOUNDARY VIOLATION`.
`AGENTS.md` §3 allows `/tmp` for throwaway checks; a copy of the worktree under `/tmp` would let `--out` sit under its
`data/`. Please say what you ran and what you only read.

## What was run (VM 101, CPU only, `CUDA_VISIBLE_DEVICES=""`; suites one at a time, foreground; `taskset -c 3-5 nice -n 10`; `python3 -u -B`; `PYTHONDONTWRITEBYTECODE=1`)

| Suite | Result |
|---|---|
| `tests/test_ch14_diag_wave1.py` (pytest, `--basetemp` under `data/`) | 37 passed |
| `tests/test_ch14_diag_wave2.py` | 37 passed |
| `tests/ch14_wave1_mutants.py --jobs 2` | 25/25 killed |
| `tests/ch14_wave2_mutants.py --jobs 2` | 36/36 killed (34/36 before two gates were strengthened: W34 and W35) |
| `test_base_revision_precondition` with `CH14_BASE_REV` = the branch commit (full) and = `7f4505a` (short) | fails on its credited tag, both |
| `tests/test_ch14_diag_land_defaults.py`, four families | **54/54 gates, 4/4 mutants** (`defaults_final`) |
| `tests/test_ll_feature_settings.py --workers 3` (it guards the optimizer) | 51/51 gates, 15/15 mutants |

**What `test_ch14_diag_land_defaults.py` does.**
- **Runs.** The real Step 5 CLI, through main's `scripts/diagnostics/e2e_step5_shim.py --split-seed 1125` (the module's
  split is unseeded), on a synthetic fixture: 240 rows, the 91 run H feature names, random values.
  - Families: lightgbm, xgboost, catboost and neural_net; 2 trials, 2 folds.
  - Each run is a fresh code tree under `--out`. The tree holds the 7 code files as at `7f4505a` (the base) or as in
    this worktree (the new code), plus symlinks for everything else. Step 5 runs with cwd = the tree.
- **Per family:** base, base again (the control), new with no level, new with `legacy`, new with `full`.
- **Comparison.**
  - JSON is compared with wall-clock keys and wall-clock run ids scrubbed.
  - Optuna studies are compared by their trials.
  - Files that differ in the control are compared by key structure only, with the NN's `/hyperparameters` masked.
  - The NN child command lines are compared as described in the NN section.
- **Health check:** `7f4505a`'s check against the new one, on the same reports and on a fixed synthetic NN report.
- **Mutants:** S01 legacy treated as full; S02 a training change; S03 legacy rules silently fixed; S04 the NN child
  gets a flag at legacy.
- **Final gate:** the worktree's `git status --porcelain --ignored`, before and after.

## Known limits (stated here so you do not have to find them)

1. **The NN is not reproducible run to run on CPU, even at `7f4505a`.** No torch seed is set on the Step 5 path; the 4
   trial calls are identical, but their scores differ, so which trial wins is random. Two base runs chose a 3-layer and
   a 4-layer winner. On the CLI the NN is checked by structure and by its child command lines. Its values are checked
   in-process with torch seeded (`test_nn_legacy_report_identical_to_base`, and the bit-identity tests). A value change
   only on the CLI path, between the child's argv and the trainer, is not covered.
2. **CatBoost's `best_model.cbm` differs between two identical `7f4505a` runs** (with `catboost_info/*`). It is compared
   by presence only. Whether its predictions differ was not measured.
3. **The defaults suite runs on a synthetic fixture.** On it, the tree families write no report at legacy, so their
   health result is `absent` under every setting. The `fixed` flip is shown on the NN's real report and on the fixed
   synthetic report only.
4. **`test_ll_feature_settings` gave 51 gates; `7cac105` records 53.** [UNVERIFIED] reading: the 2 optional
   `--runh-root` gates read the e2e-map worktree and were not run.
5. **A harness incident during the work** (READY note, "Incident"):
   - one tracked file and four ignored files were written in the `ch14diag-land` worktree through a symlink;
   - it was caught by the suite's gate and undone, and `build_tree` was fixed;
   - the main tree and the other worktrees were unaffected, with proof in the note.
6. **No GPU run and no run on real data.**

**Owner items, not in this package (for completeness):**
1. the CD-07 WATCHER line (`agents/watcher_agent.py`, hand-landed; draft diff #17, not applied);
2. a Step 5 manifest entry for `--diagnostics-level` (`agent_manifests/reinforcement.json`, hand-landed; not built);
3. the policy flips in `watcher_policies.json` (`grading_rules: fixed`, `early_stop_round_base: 1`), per the owner's
   11-23 ruling, each with his yes at landing.

## Questions

**Q1. Default unchanged.** With no `--diagnostics-level` (or `legacy`) and no `grading_rules` key (or `legacy`), do
Step 5 on these files and `training_health_check.check_training_health()` behave exactly as at `7f4505a`? That covers
the CLI, the WATCHER route as the manifests are today, the models, sidecars and Optuna studies, the reports, the NN
child command line and the health result.

**Q2. The NN gate, equal by structure only.** For the neural net, the landing gate claims structure equality, exact
child command lines (the final call's `--params` only required to be one of the run's trial proposals), and the health
result. Value equality rests on the seeded in-process tests. Is that sufficient to call the NN path "unchanged at
defaults"? Or is there a realistic way a change in these files could alter the NN's trained values on the CLI path
while passing both? Name it, if so.

**Q3. Merge.** Is the optimizer merge correct where main's LL-3 and the branch both touch the file? In particular the
argparse block (`meta_prediction_optimizer_anti_overfit.py` around `:3144-3270`), the NN subprocess command
(`:2610-2650`, diagnostics flags `:2638-2643`) and the inline tree-report call sites (`:1881`, `:2859`; helper `:2782`).

**Q4. Gates and isolation.** Does each default-unchanged gate fail when it should? That covers the precondition gate,
the 10 added mutants, the two strengthened gates (W34, W35) and S01-S04. Can any still pass vacuously? Does any test
still read another worktree, or write outside its `--out` / `--basetemp`?

**Q5. Dark landing of the `fixed` path.** The fixed rules change WATCHER's retry/skip decisions after Step 5. Are the
stale-report guard (CD-07), the skip expiry (CD-09) and the history prune (CD-10) safe to land default-off, before the
owner flips them? That is: nothing reads them at legacy, and an unknown value fails closed.

Please label every objection `LIVE` or `THEORETICAL` as `AGENTS.md` §5 requires.
