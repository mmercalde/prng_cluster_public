# READY — landing prep: ch14-diag (`1ba3d6f`) onto main `7f4505a` (TASK_CH14_DIAG_LAND_PREP_20261125)

Worktree `~/tfm_worktrees/ch14diag-land`, detached at `7f4505aaccebdd284393eb3565e75d09d1c0f3b4`. Prepared, not
landed: no commit, no push. CPU only (`CUDA_VISIBLE_DEVICES=""`), no card, no rig.

**Main has since moved to `579dbb7`** (= origin = public; CLAUDE.md, the leads register docs, and
`agents/registry/parameter_registry.py`). `git diff --name-only 7f4505a 579dbb7 -- <the 17 files>` is **empty**: none of
this package's files changed, so the three-way merge and every result below stand on `579dbb7` as well. The tests' base
pin stays `7f4505a`; none of the 7 code files it reads changed after it.

## After Beta R3 (2026-11-25; lead TASK_CH14DIAG_R3_Q5_FIX_20261125, 23:06) — read this first

**Beta X-CH14-DIAG R3** (`docs/exchange/from_beta/X-CH14-DIAG_R3_BETA_REVIEW.md`, final round):
- Q1, Q2, Q3, Q4, Q6 and Q7 are **NO OBJECTION**: the shipped code is accepted, including the lead's CD-09 review.
- **Q5 is OBJECTION · LIVE**, test isolation only: "The single-part suite writes outside `--out` in its own control and
  the isolation gates can still report success after an overwrite in a sibling directory."
- The lead accepts it as real and gave a GO, tests only.
- **No shipped file changed.** The seven shipped sha256s are identical before and after this work
  (`ch14diag-land2/data/ch14_land2/shipped_sha_before_q5.txt`).

### What changed (only `tests/test_ch14_diag_land_defaults.py`)

| Beta R3 Q5 point (`:99-101`) | Change | Evidence (`--part single` 86/86 · 9/9; `--part compare` 29/29 · 2/2) |
|---|---|---|
| A LightGBM import can create a matplotlib cache under the inherited `TMPDIR`, outside `--out` | `redirect_env_roots`: `TMPDIR`, `XDG_CACHE_HOME`, `MPLCONFIGDIR` and `CUDA_CACHE_PATH` under `--out/_env/`, set before any third-party import. Every child inherits them (all subprocesses start with `env=dict(os.environ, …)`). The pattern of main's `tests/test_ll6_inference_bundle.py:85-100` | `_env/{tmp,cache,mpl,cuda_cache}` exist under each `--out`; no `/tmp/matplotlib-*` before or after any suite in this round |
| The snapshot pruned `data/`, so a sibling overwrite under `data/` passed both gates | `boundary_snapshot`: every file under `data/` with **only `--out` pruned**, as (size, mtime_ns, sha256), symlinks by target and never followed; every file of the worktree outside `data/` and `.git`, the same signature; `git status` outside `data/`; new `/tmp/matplotlib-*`; the user's matplotlib directories. Compared before and after each part: `BOUNDARY.nothing_outside_out_changed` | single: 314,261 `data/` entries and 2,409 tree files watched, changes `[]`; compare: 348,323 and 2,409, changes `[]` |
| A control that overwrites an existing sibling and fails the check | `BOUNDARY.negative_control`: overwrites an existing file under `data/` outside `--out` (same size, mtime restored, so only sha256 can see it); the gate must go **red**; then restores the bytes and mtime, and the gate must be **green** again | RED then GREEN in both parts. Target: an existing non-live file (`data/ch14_land2/shipped_sha_before_q5.txt`), restored byte-identical |
| The CatBoost control planted at the worktree root, outside `--out` | `negative_control` builds a **disposable source root inside `--out`**: links to the worktree's top level, except a real `catboost_info/` with a sentinel. `build_tree(…, src_root=…)` builds from it. RED: R1 policy, linked, written through, snapshot changed, R1 status view unchanged. GREEN: new policy, not linked, unchanged. `NC.nothing_planted_at_the_worktree_root` | all three pass; no `catboost_info/` at the worktree root |

**Also from this round:**
- **Logs inside `--out`.** On its first smoke run, the new boundary gate caught **my own `tee` log** beside `--out`, the
  same thing LL6 hit in development. The suite now accepts an `--out` that holds only `console.log`, and every run's log
  is `<out>/console.log`.
- **The overwrite target skips files modified in the last 120 s**, so a live log is never picked. The first smoke picked
  the live `tee` log, and the concurrent append failed the "restored" check. Both smoke runs are kept under
  `data/ch14_land2/q5/smoke_lgb*`.
- **Q6 (optional gate, added).** `LAND.neural_net.fixed_rules_grade_real_report_validly`: on the real NN report the
  fixed rules must return a valid grade (ok, warning or critical, either one) with no "Health check module error"
  issue. Final run: critical/RETRY, no module error.
- **Q6 wording.** Plain-summary item 6 is qualified as the synthetic-fixture result (below).
- **Finding 1 (R3).** Two governing records added to the prior record:
  - `logs/e2e_session/TASK_CH14_DIAG_LAND_PREP_20261125.txt:2-6` ("No test may read another worktree or write outside
    its --out"; full-sha baselines; a mutant per gate). This round's fix closes the two violations of it Beta found.
  - `docs/exchange/from_beta/X-E2E-GRADE_R1_BETA_REVIEW.md:55`: the stock compare route archives canonical artifacts
    after a failed child and can select by their score. The "catboost child exits 1 and its archive holds the previous
    family's files" fact reported in "After Beta R1" is therefore **ALREADY RECORDED**, not new.
- **Scope, stated.** The wave-1/2 pytest suites and the mutant runners do not redirect temp and cache roots. They write
  under `--basetemp` / `--out`, and no `/tmp/matplotlib-*` appeared during them in this round (checked before and after
  each). Beta's R3 Q5 concerned the defaults suite.

**New sha256:**

| File | Before R3's fix | Now |
|---|---|---|
| `tests/test_ch14_diag_land_defaults.py` | `80a20599dd2c0c4d562960129048c4c1978c3e3e6f40a9588537a743520ed35b` | `ebb264df1cc9cf88cadd7d2648060a33a119c0d23432a27a4280974a205ab4e5` |

The other 17 files are unchanged; this note changes too, by this section.

### Re-run in `ch14diag-land2` (one at a time; `taskset -c 3-5 nice -n 10`; `python3 -u -B`; CPU; logs `data/ch14_land2/q5/logs/` and `<out>/console.log`)

| Suite | Result |
|---|---|
| `tests/test_ch14_diag_wave1.py` | **37 passed** |
| `tests/ch14_wave1_mutants.py --jobs 2` | **25/25 killed** |
| `tests/test_ch14_diag_wave2.py` | **37 passed** |
| `tests/ch14_wave2_mutants.py --jobs 2` | **36/36 killed** |
| defaults `--part single` | **86/86 gates, 9/9 mutants** (S01-S06, V2-V4); `BOUNDARY` and its control; `NC` red, green, nothing at the root; Q6 gate |
| defaults `--part compare` | **29/29 gates, 2/2 mutants** (S07, S08); `BOUNDARY` and its control |

- The two defaults parts ran as background jobs, waited on to completion with nothing else running: 8 min 23 s and
  9 min 32 s, close to the 10-minute foreground cap.
- **Commits:** none. HEAD stays `579dbb7`, `rev-list --all` is 1022 before and after, and there is no `.git` under
  `data/` (owner ruling 19:26).
- **Main** is still `1bb75f0`; none of the 18 files differ.

## After Beta R1 (2026-11-25; lead TASK_CH14DIAG_BETA_FIXES_20261125, 20:02) — read this first

**Where the package is now:** `~/tfm_worktrees/ch14diag-land2`, a fresh detached worktree of `579dbb7` holding only
the 18 landing files.
- Every result in this section was run there.
- `~/tfm_worktrees/ch14diag-land` (`7f4505a`) is the development tree; its copies are superseded where the sha256s
  below differ.
- **Main has since moved to `1bb75f0`:** the owner's R1 exchange landing `ed2d22d`, LL-6 `c16f990`/`2384024`, the
  registry repair `b61cc06`/`7e9c5ff`/`3216bc4`, item 4 `ef272cc`, and two closing notes `05e30d2`/`1bb75f0`.
  `git diff --name-only 579dbb7 1bb75f0 -- <the 18 files>` is **empty**.

**Beta X-CH14-DIAG R1** (`docs/exchange/from_beta/X-CH14-DIAG_R1_BETA_REVIEW.md`, on main since `ed2d22d`):

| Question | Position | Severity |
|---|---|---|
| Q1 · Default unchanged | NEEDS EVIDENCE | — |
| Q2 · NN gate | NEEDS EVIDENCE | — |
| Q3 · Merge | NO OBJECTION | — |
| Q4 · Gates and isolation | OBJECTION | LIVE — the harness can overwrite an existing `catboost_info/` outside `--out` and still pass isolation |

Q5 · Dark landing: NO OBJECTION. No label is contested (lead). **Q4 is fixed below** ("Q4 · fixed").

### What changed, on the lead's GO

| Item | Change | Evidence |
|---|---|---|
| Q2 gap 1, swapped winner | New gate `R2.neural_net.<run>_final_params_equal_own_best_trial`: the final NN call's `--params` must equal the run's own `study.best_trial.params`, read with optuna from a **copy** of its study database | **S05** (Step 5 hands the final model the worst trial): the R1 child-argv gate **PASSES (missed)**; the new gate **FAILS** |
| Q2 gap 2, worker boundary | New test-only `WORKER_DRIVER` (in `tests/test_ch14_diag_land_defaults.py`) runs the real `train_single_trial.main()` on Step 5's own final command line and a fixed NPZ. It records what `main()` hands `train_neural_net`: params, flags, kwargs, and the sha256/shape/dtype of the four arrays. Base and new are compared. Rule stated in the code: the one kwarg this landing adds, `diagnostics_options`, counts as absent when `None` (its default) | **S06** (the worker multiplies the learning rate by 1.5 after `json.loads(args.params)`, Beta's case): the R1 structure gate and the R1 child-argv gate both **PASS (missed)**; the new boundary gate **FAILS** (lr 1.5×) |
| Q2 gap 3, values | The same driver seeds torch/numpy/random **itself**; the worker is not changed. Weights (sha256 per tensor) and predictions on `X_val` (rebuilt from the checkpoint) are compared with base | **Control:** base seeded twice is identical (inputs, weights, predictions), so the check is real. Base = new at legacy. `full` changes only the inputs record (it passes `diagnostics_options`), not weights or predictions. Killed by S06 |
| Q1, WATCHER's route | `--part compare`: `--compare-models --enable-diagnostics` through the real CLI. Base, base again, default, `legacy` and `full` are compared on: the children's command lines; the summary's per-family return codes and archived files; outputs by content where deterministic, else by structure; the CD-06 pointer only at `full`; the health check called as WATCHER calls it; and a per-run rule that the top-level artifacts are the run's **own** winner's, restored | **S07** (the compare child gets an extra flag at legacy): the R1 single-family gates **PASS (missed; marker hits 0, not on that path)**; the compare gate **FAILS**. **S08** (the restore copies another family's artifacts): the winner-restore gate **FAILS** |
| Whitespace | The six whitespace-only lines `git diff --check` flagged stripped (`meta_prediction_optimizer_anti_overfit.py` `:3269/3271/3273`, `training_health_check.py` `:381/428/1556`) | `diff -w` against the reviewed files is **empty**; plain diff is those 6 lines only (reviewed copies kept in `ch14diag-land/data/ch14_land/reviewed_r1/`) |
| Q5 wording | Corrected below | — |
| Finding 1 | Added to the prior record; three questions below | — |

**Old → new sha256** of the changed files (the two test files changed again in the Q4 fix, below):

| File | Reviewed in R1 | Now |
|---|---|---|
| `meta_prediction_optimizer_anti_overfit.py` | `48052f2021ee055e834852b64d04497caa56b542383779d824c9e3e96dbd0e17` | `ab836d4283414f791fab0a0d3c609f40020e8eedade1ddd77f219c26535d8b49` |
| `training_health_check.py` | `ac8ee08ece13f2b7517977287b530376e20680c5ce5d74dc1f5aa25ccc744ad0` | `fabb6490a21ea0b89d330e38b9fe280a582492d2a520b34e1bbc47680fb8cb9e` |
| `tests/test_ch14_diag_land_defaults.py` | `03e67c3f6c4ec64a2637d3f6b67259458be64f981f4ba7db55834e59fc72607f` | `484d1167a4cb1328ad584714618bb14194b273cb0496526ecd06644b410784ba` |

The other 15 files are byte-identical to the versions R1 reviewed. This note changes too, by this section.

### Facts found while building the compare part (measured, not changed)

- **The catboost compare child exits 1 at `7f4505a` too.** It is identical in base and new, and recorded in the summary
  digest. Its archive folder then holds the leftover artifacts of the family before it (xgboost's). That is why the
  first S08 (restore "catboost") was a no-op when xgboost won; S08 now uses lightgbm, or the NN when lightgbm won.
- **Compare children are unseeded processes.** Main's shim seeds only the parent. So scores and the winner vary run to
  run, even at `7f4505a`: two base runs and three new runs picked lightgbm or the NN in different attempts. 23 output
  files differ between two base runs. Those files are compared by structure. The winner-dependent top-level
  `best_model.*` files are checked per run, by the restore rule, not across runs. Attempt 1 failed exactly there, on
  `models/reinforcement/best_model.meta.json` (base's winner lightgbm, new's the NN). The rule was fixed and checked
  offline on attempt 1's data before the re-run. Attempts 1 and 2 are kept:
  `data/ch14_land2/defaults_compare_attempt{1,2}`.

### Results in `ch14diag-land2` (each suite once, alone; `taskset -c 3-5 nice -n 10`; `python3 -u -B`; `PYTHONDONTWRITEBYTECODE=1`; CPU)

| Suite | Result | Wall | Log (`ch14diag-land2/data/ch14_land2/logs/`) |
|---|---|---|---|
| `tests/test_ch14_diag_wave1.py` | **37 passed** | 39 s | `w1_pytest.log` |
| `tests/test_ch14_diag_wave2.py` | **37 passed** | 12 s | `w2_pytest.log` |
| `tests/ch14_wave1_mutants.py --jobs 2` | **25/25 killed** | 43 s | `mutants_w1.log` |
| `tests/ch14_wave2_mutants.py --jobs 2` | **36/36 killed** | 38 s | `mutants_w2.log` |
| precondition mutant (`CH14_BASE_REV` = branch full sha; = `7f4505a` short) | **both fail on `[CREDIT:BASE_PRECONDITION]`** | <1 s | `precond_mutant_*.log` |
| `tests/test_ch14_diag_land_defaults.py --part single` | **64/64 gates, 6/6 mutants (S01-S06)** | 7 min 19 s | `defaults_single.log`, `../defaults_single/summary.json` |
| `tests/test_ch14_diag_land_defaults.py --part compare` | **25/25 gates, 2/2 mutants (S07, S08)** | 9 min 18 s | `defaults_compare.log`, `../defaults_compare/summary.json` |
| `tests/test_ll_feature_settings.py --workers 3` | **51/51 gates, 15/15 mutants** | 1 min 26 s | `ll_feature_settings.log` |

The compare part ran as a background job (9 min 18 s exceeds the 10-minute foreground cap with margin too thin). It was
waited on to completion, and no other suite ran meanwhile.

**Commits (owner ruling `logs/e2e_session/OWNER_RULING_fixture_commits_20261125.txt`, 19:26):** no suite here runs
`git commit`.
- `ch14diag-land2`'s HEAD is `579dbb7` before and after.
- No `.git` directory exists anywhere under its `data/`.
- `git rev-list --all --count` went 1013 → 1015. The two new commits are the owner's LL-6 landings on main (`c16f990`
  20:40:30, `2384024` 20:41:42, author mmercalde, refs `main`/`origin/main`/`public/main`); no other ref moved.
- **Writes:** the worktree's `git status --porcelain --ignored` changed only under `data/`.

### Q5 · corrected wording (Beta R1 `:99`)

My R1 request said the new settings are not read at legacy and that "an unknown value fails closed". Both were
inexact. As measured by Beta:
- the settings **are read and validated at legacy** (`training_health_check.py:130-178`, `:242-250`);
- an unknown value is rejected for grading and its side effects;
- `check_training_health()` then returns `PROCEED_WITH_NOTE` with `severity` `absent` and a module-error note, and
  **WATCHER carries on** (`training_health_check.py:223-239`, `agents/watcher_agent.py:2570-2586`);
- in the compare optimizer, an invalid `expiry_mode` with a registry entry raises before dispatch.

### Finding 1 · the 11-24 owner ruling on benching (added to the prior record)

`logs/e2e_session/OWNER_RULINGS_pending_list_20261124.txt:20-33`:
- **10:47-10:50 RULED:** "a poor health grade never removes a model family from the comparison; it triggers a
  why-report. Benching stays in the code as a switch, off. … Flips when landed."
- **TODO-WHY-1** (`:27-28`): the automatic trigger and queue are deferred; the lead runs the attribution cycle by hand.
- **11:11 RULED #2** (`:31-33`): "the lead decides the four housekeeping items (archive move not delete; working
  dead-unit measure; training health log on; Chapter 13 loads the model as Step 6 does once it passes on a real
  winner). Each flips only when landed."

**Status in this package (ALREADY RECORDED, not changed):**
- today's benching (the skip after `max_consecutive_critical`, `meta_prediction_optimizer_anti_overfit.py:148-188`,
  `training_health_check.py:1028-1076`) is **kept as it is**; Beta confirmed it at count three (R1 `:97`);
- **no off switch for benching is built here**;
- every setting stays at today's value.

**Questions, undecided** (asked as the lead directed). The record above already gives these to the lead, with a
direction, each flipping at landing; I have not decided between the record and the question.
1. CD-10 history prune, when it is turned on: move the oldest files (today's draft default) or delete them?
2. CD-23 dead-unit metric: keep `linear_zero` (today's) as the default, or default to `always_negative_preact`?
3. CD-11 training health log: off (today's default) or on, so the Strategy Advisor has something to read?

### Q4 · fixed (lead TASK_CH14DIAG_Q4_FIX_20261125, 21:32: "accepted as real; label not contested; GO")

**Test-side only; no shipped file changed.** Only `tests/test_ch14_diag_land_defaults.py` and
`tests/test_ch14_diag_wave1.py` changed.

**1. CatBoost's train dir is always under `--out`.**
- No shipped file passes `train_dir`, so CatBoost creates `catboost_info/` in its process's cwd.
- Every CatBoost process these tests start already ran with its cwd at a run tree under `--out`: the Step 5 parent,
  each compare child (the shim records `cwd` = the run tree), and the S/V mutant runs.
- The defect was that the tree could hold a **link** called `catboost_info`. `build_tree` now links **only entries git
  tracks at HEAD**. An ignored or untracked entry (such as a pre-existing `catboost_info/`) is never linked.
- Output places (`OUTPUT_TOP`: `data`, `models`, `diagnostics_outputs`, `optuna_studies`, `agents`, `logs`,
  `catboost_info`, …) are never linked.
- Tracked root `.jsonl` logs (`acceptance_decisions.jsonl`, `chapter13_cycle_history.jsonl`,
  `watcher_training_incidents.jsonl`) are **copied**, so an append lands in the copy.
- R1's skip-list behaviour survives only as `link_policy="legacy"`, used by the negative control.

**2. The isolation gate sees existing directories, not only new paths.**
`LAND.no_existing_dir_outside_out_changed` snapshots (path, size, mtime_ns, sha256) of every file in every existing
`catboost_info/`:
- anywhere in the worktree outside `data/` and `.git`;
- directly in each parent of the worktree up to the home directory;
- in the main checkout (depth 4).

It takes them before and after each part, symlinks never followed. On this box the only one is
`~/distributed_prng_analysis/catboost_info` (7 entries): unchanged in both parts.

**The negative control** (`NC.*`, single part) plants a sentinel `catboost_info/` at the worktree root, outside
`--out`, only if none exists. It then runs one CatBoost Step 5:
- **RED** under the R1 tree policy: the tree links it, CatBoost writes through it (`catboost_training.json`,
  `learn_error.tsv`, `learn/` and `test/` event files changed), the new gate turns red, and **the R1 git-status gate
  stays unchanged (missed)**, because the directory is ignored.
- **GREEN** under the new policy: not linked, byte-identical.
- **Cleanup:** the planted directory is removed. The worktree's status outside `data/` is the same as before the suite.

**3. The four vacuity points, each with a mutant the R1 gate missed:**

| Point (Beta R1 `:91`) | New gate | Mutant: R1 gate → new gate |
|---|---|---|
| V1: the base-pin precondition passes at the full `41ad509` sha | The check also requires main's LL-3 marker (`--exclude-feature-groups`) and no `--diagnostics-level` in the base optimizer. It now runs **inside the `head_dir` fixture**, so every identity test fails on a wrong pin (`tests/test_ch14_diag_wave1.py`, `_base_precondition`) | Pin = full `41ad509`: the R1 check and `test_legacy_rules_identical_to_base[late_best]` **pass (missed)**; the new check and the same identity test **fail** on `[CREDIT:BASE_PRECONDITION]` (pytest reports a fixture error). Log `q4/logs/v1_pin_41ad509.log` |
| V2: two empty child traces count as equal | `LAND.<fam>.<run>_child_call_count`: NN exactly 5 children (2 trials × 2 folds + 1 final); trees 0; compare 4 | V2, both traces empty: R1 child-argv gate **PASS (missed)**; count gate **FAIL** |
| V3: a difference anywhere counts as "full changes the reports" | `nonvacuity_full_changes_reports` needs a changed file **under `diagnostics_outputs/`** | V3, the inline tree report written to the run root at full: R1 "any difference" gate **PASS (missed)**; new gate **FAIL** |
| V4: tree `fixed_changes_health` rows pass on dict differences | Trees: both rules must give exactly absent/PROCEED (expectation, not a change claim). Fixed synthetic report: exactly critical/RETRY → ok/PROCEED. **The NN's real-report row is no longer a gate:** attempt 1 measured fixed = critical/RETRY on a real early stop, so a change there is data-dependent. Its outcome is recorded in `summary.json` | V4, the fixed path grades with the legacy rules: R1 dict-inequality row **PASS (missed)**; fixed-report expectation **FAIL** (critical/RETRY) |

**Harness fixes during this work (disclosed):**
- **Single attempt 1** (`q4/defaults_single_attempt1`, 81/83) failed two gates.
  - The NN real-report row (above) failed.
  - `NC.red` reported the R1 status gate as "changed". That came from its own run's files under `data/`: the control
    compared raw `git status`, while the R1 gate ignores `data/`. Fixed to compare as the R1 gate does.
  - The write-through and the new gate's detection were real in attempt 1.

**New sha256 of the two test files:**

| File | Before the Q4 fix | Now |
|---|---|---|
| `tests/test_ch14_diag_land_defaults.py` | `484d1167a4cb1328ad584714618bb14194b273cb0496526ecd06644b410784ba` | `80a20599dd2c0c4d562960129048c4c1978c3e3e6f40a9588537a743520ed35b` |
| `tests/test_ch14_diag_wave1.py` | `5fff62b4ee9eaf87184ae85d2bad7e9e01283b3a6db0e7181d11ea31727aac91` | `3996556ee7c877e64411f9430b14e09f2cd6da4ae56ca4be2fc8ffe8096074cd` |

**Re-run of the affected suites** (`ch14diag-land2`, one at a time, `taskset -c 3-5 nice -n 10`; logs in
`data/ch14_land2/q4/logs/`):

| Suite | Result |
|---|---|
| `tests/test_ch14_diag_wave1.py` | **37 passed** |
| `tests/ch14_wave1_mutants.py --jobs 2` | **25/25 killed** |
| V1 pin check (R1 vs new, pin = full `41ad509`) | R1 passes (missed); new fails, 2 credited |
| defaults `--part single` | **82/82 gates, 9/9 mutants** (S01-S06, V2-V4), NC red/green/cleanup |
| defaults `--part compare` | **26/26 gates, 2/2 mutants** (S07, S08) |

- **Not re-run, unaffected:** wave 2 (37/37), its mutants (36/36) and `test_ll_feature_settings` (51/51, 15/15) from
  the post-R1 run. Their test files and the code are unchanged.
- **Commits:** none by any suite. The worktree's HEAD stays `579dbb7` and there is no `.git` under `data/`.
  `rev-list --all` rose 1019 → 1022 only by the owner's main commits `ef272cc`, `05e30d2`, `1bb75f0`.
- **Main is now `1bb75f0`;** it touches none of the 18 files.

## Prior record

**Searches run** (main `7f4505a`):
- `git grep -l -E "CH14-DIAG|diagnostics-level|grading_rules"` over `docs/PROJECT_FILE_CATALOG.md docs/TB_RULING_*
  docs/PROPOSAL_* docs/CHAPTER_* docs/*_AUDIT.md docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/LEADS.md docs/exchange/`
  → **no hits**.
- `ls docs/exchange/{to,from}_beta | grep -iE "ch14|diag"` → **no Beta thread** for this work.
- `/bin/grep -lE "ch14-diag|CH14-DIAG|ch14_diag" logs/e2e_session/*.txt` → `TASK_CH14_DIAG.txt`, `TASK_LOOP_DIAG.txt`,
  `TASK_CH14_DIAG_LAND_PREP_20261125.txt` (this task), `TASK_ITEM5_SURVEY_REST_20261125.txt`,
  `TASK_CH14_LEARN_SURVEY_20261125.txt`, `TASK_LL_SWITCHES_BETA_20261125.txt`, `TASK_LOOP_SELFPLAY.txt`.
- `git diff --name-only 41ad509 1ba3d6f`, then `git log 41ad509..7f4505a -- <file>` for each file.

**Documents found that decide, schedule or record the subject:**
- `~/tfm_worktrees/e2e-map/docs/drafts/loop_v1/SURVEY_ITEM5_BRANCHES_20261125.md` §1 (read in full): ch14-diag holds 16
  files, all class (c); only the optimizer changed on main since `41ad509` (`7cac105`, LL-3). The survey names the re-pin,
  the manifest re-point, and the WATCHER/manifest parts as the owner's.
- The branch's own notes, read in full: `READY_LOOP_DIAG_20261123.md` (wave 1) and `READY_LOOP_DIAG_2_20261123.md`
  (wave 2), and the plan `docs/drafts/ch14_v1/PLAN_CH14_DIAGNOSTICS.md` (prior record `:13-96`).
- `logs/e2e_session/OWNER_RULING_defaults_20261123.txt` (19:23): fixed rules, round base 1, and `full` by default once
  the manifest entry exists. The condition: "each default flips only when that piece of work is landed, which still
  needs his yes at the time".
- `logs/e2e_session/TASK_CH14_DIAG_LAND_PREP_20261125.txt` (this task).

- **Added after Beta R3 (Finding 1):** `logs/e2e_session/TASK_CH14_DIAG_LAND_PREP_20261125.txt:2-6` (no test writes
  outside its `--out`; full-sha baselines; a mutant per gate) and `docs/exchange/from_beta/X-E2E-GRADE_R1_BETA_REVIEW.md:55`
  (a failed compare child's archive holds canonical leftovers; ALREADY RECORDED). See "After Beta R3".
- **Added after Beta R1 (Finding 1):** `logs/e2e_session/OWNER_RULINGS_pending_list_20261124.txt:20-33` — model benching
  (a poor grade never removes a family; benching stays as a switch, off, flips when landed), TODO-WHY-1, and the four
  housekeeping items given to the lead. See "After Beta R1", Finding 1.

**Labels:**
- The work itself, its settings and its defaults: ALREADY RECORDED (the two READY notes; the survey §1).
- The re-pin to `7f4505a`, the new Step 5 default gate, the 10 added mutants and the harness fixes below: **NEW**.
- CHANGES A RULING: none. Every default stays at today's behaviour; the owner's 11-23 defaults are not flipped.

**Not searched:** the host outside the named paths, ser8 archives, the private repository, project files outside the
repository.

## Plain summary

1. Clean 3-way merge of all 16 files; 15 equal the branch, the optimizer = main's LL-3 + exactly the branch's delta.
2. No LL-6 file (`prediction_generator.py`, `chapter_13_orchestrator.py`, `models/inference_bundle.py`) touched or imported.
3. Every baseline re-pinned to full sha `7f4505a…`; a new precondition gate fails on a short or already-merged pin.
4. Defaults change nothing: real Step 5 CLI, all four families, seeded split; trees content-equal to `7f4505a`.
5. The NN is equal by structure only: it is not reproducible run to run on CPU, even at `7f4505a`.
6. Health check at legacy = `7f4505a`'s on the same report. On the fixed synthetic NN report, `fixed` gives ok/PROCEED where
   legacy gives critical/RETRY; on the real NN report the outcome varies run to run and is recorded (Beta R3 Q6).
7. Tests: wave 1 37/37, wave 2 37/37, mutants 25/25 and 36/36, landing gate 54/54 (4/4), LL 51/51 (15/15).
8. Nothing ignored, governed or hand-landed; the WATCHER line, manifest entry and policy flips stay the owner's.
9. Pools and the k/1000 measurement: nothing at defaults; `fixed` would change WATCHER's retry/skip after Step 5.
10. One harness incident in this scratch worktree (a model file written through a link), caught, undone; main untouched.

## Files (17), sha256 of the version in this worktree

| File | N/M on `7f4505a` | sha256 | Class |
|---|---|---|---|
| `meta_prediction_optimizer_anti_overfit.py` | M (3-way merge) | `ab836d4283414f791fab0a0d3c609f40020e8eedade1ddd77f219c26535d8b49` | **pipeline (Step 5)** (after R1: whitespace stripped) |
| `train_single_trial.py` | M | `8db47e737183cdcadd0fa698267b583566595eaa09d4a5c4ea3e111414a793b2` | **pipeline (Step 5 trial worker)** |
| `training_diagnostics.py` | M | `bd3787d7a61ccb4604ca4fa2537503bf42fc4dabc01ceaf3fcc4907de9852541` | Step 5 report producer |
| `training_health_check.py` | M | `fabb6490a21ea0b89d330e38b9fe280a582492d2a520b34e1bbc47680fb8cb9e` | WATCHER post-Step-5 health path (after R1: whitespace stripped) |
| `diagnostics_llm_analyzer.py` | M | `cb73536dbe0ce6dc11b1b59b6db1bca9c6bfa70858700258d4abfb80ad6801c1` | Ch14 LLM bundle |
| `parameter_advisor.py` | M | `d9bf73dc82af7c640069956ff97edf8d349f44dceeee1573e9c074895cfd17d2` | advisor reader |
| `agents/contexts/advisor_bundle.py` | M | `95711636feb165781332629a14a15a02b7c53b765c73629b261a8675dd5eb63b` | agents/ (not hand-landed) |
| `tests/test_ch14_diag_wave1.py` | N (re-fit) | `3996556ee7c877e64411f9430b14e09f2cd6da4ae56ca4be2fc8ffe8096074cd` | test (Q4: base pin in the fixture) |
| `tests/test_ch14_diag_wave2.py` | N (re-fit) | `81749ce93235b98dcc3a70563726a299b4b3264c6054d373666da1325bd30cf9` | test |
| `tests/ch14_wave1_mutants.py` | N (re-fit) | `390bcf4845e205e39aa5e2934e018b527212ce1fd37f187ab25dde0979459944` | test |
| `tests/ch14_wave2_mutants.py` | N (re-fit) | `37fdc91fdd87f0d2b9f1616c2d0de5a348927f3a874439a8fc8e255d8376f4b7` | test |
| `tests/ch14_cpu_patch.py` | N | `5e5fc75666da75791c29ff5100b383ea83a6180505e8524d8f6caa1990e9b504` | test |
| `tests/test_ch14_diag_land_defaults.py` | **N (new, this task)** | `ebb264df1cc9cf88cadd7d2648060a33a119c0d23432a27a4280974a205ab4e5` | test (after R1: gates (1)-(4), S05-S08; Q4 fix; R3 Q5 fix) |
| `docs/drafts/ch14_v1/PLAN_CH14_DIAGNOSTICS.md` | N | `9b74b0aee7a2bf577fdafb42e9bad05f028cbcd6c376c6803d26d9aea9cfd118` | doc |
| `docs/drafts/loop_v1/READY_LOOP_DIAG_20261123.md` | N | `4ae219bb32fb5ac8eb4f536c2a074710c1630687a6f89290d94c61c34654f3ac` | doc |
| `docs/drafts/loop_v1/READY_LOOP_DIAG_2_20261123.md` | N | `5d13a3f9d70e8f55918f698c1fbeb0f41464c1b7b74f17d472b89751a5cd7a9a` | doc |
| `docs/drafts/loop_v1/DRAFT_WATCHER_CD07_20261123.diff` | N | `0cfec893a039784f4dffb5e84047bd7213e52be0d78c5b5df1db64236729c398` | doc: a draft diff **for `agents/watcher_agent.py` (hand-landed); not applied** |

This note is the 18th file (`docs/drafts/loop_v1/READY_LAND_CH14_DIAG_20261125.md`).

**Checked for every file:** `git check-ignore --no-index` → none is ignored. None is in `GOVERNED_FILES`
(`scripts/gate12_parity_gate.py:174-185`), nor in `HAND_LANDED_FILES`/`HAND_LANDED_DIRS` (`scripts/owner_land.py:86-87`).
**Ignored or hand-landed files the owner must add by hand: none.**

**Owner items, not in this package:**
- the CD-07 WATCHER line (draft diff above, `agents/watcher_agent.py`);
- the Step 5 manifest entry for `--diagnostics-level` (`agent_manifests/reinforcement.json`, proposed in
  `READY_LOOP_DIAG_2`; not built);
- the policy flips in `watcher_policies.json` (`grading_rules: fixed`, `early_stop_round_base: 1`), per his 11-23
  ruling, each with his yes when it lands.

## The merge

- **15 files:** `git merge-file` with base `41ad509` gives exactly the branch blob, because main has not touched them.
- **The optimizer:** merged cleanly (rc 0).
  - Merged versus branch = exactly main's `41ad509→7f4505a` delta (90 diff lines, LL-3).
  - Merged versus main = exactly the branch's delta (118 diff lines).
  - Both checked by sorted diff-line comparison; the file parses.
- **The branch's code imports none of the LL-6 files** (grep over the branch's changed `.py` files).

## Test re-fit (what changed in the tests, and why)

- **Pins:** `HEAD_REV` = `7f4505aaccebdd284393eb3565e75d09d1c0f3b4` in both suites (was `41ad509`).
  - Three tests were renamed `…_identical_to_41ad509` → `…_identical_to_base`, and the mutant runners updated.
  - `CH14_BASE_REV` overrides the pin in wave 1 only, so the precondition gate can be shown to fail.
- **Writes:**
  - The base-revision copies now go under pytest's `--basetemp` (they were written to `data/ch14/head_*`).
  - The mutant runners take `--out` as a new directory under `data/` and write everything there (they wrote to
    `data/ch14/...` beside a JSON `--out`).
- **No other worktree is read:** wave 2's manifest test reads the repo's `models/feature_manifest.json` (blob
  `a8f67496`, the same as the ch14-learn copy). Its `skipif` is replaced by an assertion that the file exists.
- **New `test_base_revision_precondition`:**
  - the pin is a full sha that resolves;
  - the base files lack `DIAGNOSTICS_LEVELS` / `grading_rules` / `--diagnostics-level`.
- **Credit tags added** where a gate had none: `CD06_ABSENT`, `CD07_STALE_ACTION`, `CD07_FAIL_CLOSED`.
- **Two gates strengthened after a mutant survived:**
  - W34: the FIFO default gate also asserts that nothing is moved or written outside the folder.
  - W35: the fallback assert uses `.get`, so it fails on its tag instead of raising `KeyError`.
- **10 mutants added**, one for each gate that had none:
  - M22, M23, M24;
  - W30 to W36.
  - `test_unverified5_…` checks a third-party library fact (xgboost/lightgbm Boosters), not code under test; it has no
    mutant.
- **New `tests/test_ch14_diag_land_defaults.py` (task item 3):**
  - **Step 5 runs.** Each run is a fresh code tree under `--out`: symlinks into the worktree, except the 7 code files,
    taken from `7f4505a` or from this worktree. Step 5 runs with cwd = that tree, as production runs from the repo root
    (the final NN subprocess is started as a relative `train_single_trial.py`).
  - **Split seed.** Runs go through main's `scripts/diagnostics/e2e_step5_shim.py --split-seed 1125`, because the
    module's split is unseeded (TODO-E2E-21).
  - **Plan per family:** base, base again (control), new with no level, new with `legacy`, and new with `full`.
  - **Comparison.**
    - JSON is compared with wall-clock keys and wall-clock run ids scrubbed.
    - Optuna studies are compared by their trials (read `mode=ro&immutable=1`).
    - Files that differ in the control are excluded from the value comparison and compared by key structure instead.
  - **Health check.** `7f4505a`'s check and the new check run on the same report (the base run's), with no key,
    `legacy`, and `fixed`.
  - **Mutants:** S01 legacy treated as full; S02 a training change; S03 legacy rules silently fixed.
  - **Final gate:** `git status --porcelain --ignored` of the worktree, before and after.

## Results (each suite once, alone, foreground; `taskset -c 3-5 nice -n 10`, `python3 -u -B`, `PYTHONDONTWRITEBYTECODE=1`, venv `~/venvs/torch`)

| Suite | Result | Wall | Peak RSS | Log (`data/ch14_land/logs/`) |
|---|---|---|---|---|
| `tests/test_ch14_diag_wave1.py` | **37 passed** | 35 s | 723 MB | `w1_pytest.log` |
| `tests/test_ch14_diag_wave2.py` (after the W34/W35 gate fixes) | **37 passed** | 12 s | 643 MB | `w2_pytest.log` |
| `tests/ch14_wave1_mutants.py --jobs 2` | **25/25 killed** | 43 s | 559 MB | `mutants_w1.log`, `../mutants_w1/summary.json` |
| `tests/ch14_wave2_mutants.py --jobs 2` (re-run after the fixes) | **36/36 killed** | 37 s | 567 MB | `mutants_w2.log`, `../mutants_w2/summary.json` |
| precondition mutant (`CH14_BASE_REV` = `1ba3d6f…` full; and `7f4505a` short) | **fails on `[CREDIT:BASE_PRECONDITION]`, both** | <1 s | — | `precond_mutant_*.log` |
| `tests/test_ch14_diag_land_defaults.py` (all four families), **final** | **54/54 gates, 4/4 mutants** | 318 s | 653 MB (children 638 MB) | `defaults_final.log`, `../defaults_final/summary.json` |
| (the same suite, before the NN child-argv gate, the fixed-report health gate and S04 were added) | 42/42 gates, 3/3 mutants | 292 s | 655 MB | `defaults_all3.log` |
| `tests/test_ll_feature_settings.py --workers 3` | **51/51 gates, 15/15 mutants** | 86 s | 547 MB | `ll_feature_settings.log` |

**First result of the wave-2 mutants, before the fixes:** 34/36. W34 survived (rc 0). W35 failed with an uncredited
`KeyError`.

**The defaults suite, per family (`defaults_all3`):**

| Family | Excluded by the control (differs between two `7f4505a` runs) | Compared by content | Default and `legacy` = base | `full`: training equal, reports differ | Health (base / new legacy / new fixed) |
|---|---|---|---|---|---|
| lightgbm | none | 3 files (model, sidecar, study trials) | yes | yes / 2 reports | absent / absent / absent |
| xgboost | none | 3 files | yes | yes / 2 reports | absent / absent / absent |
| catboost | 5: `catboost_info/*` and `best_model.cbm` | 4 files, plus 5 by structure | yes | yes / 2 reports | absent / absent / absent |
| neural_net | 5: both reports, `best_model.pth`, the sidecar, the study | 0 by content; 5 by structure | yes (structure) | yes / 2 reports (structure) | **critical/RETRY / critical/RETRY / ok/PROCEED** |

**Mutants:**
- S01 changed both lightgbm reports.
- S02 changed the lightgbm model, sidecar and study.
- S03 turned the NN health result into ok/0 issues, against critical/1 issue.

**Why the suite grew after `defaults_all3` (2026-11-25 19:10 task):**
- the NN child-argv gate (below, NN section);
- a health gate on a fixed synthetic report;
- mutant S04.

The intermediate runs were `defaults_all4` (S03 survived: it depended on the NN's random report; S04 invalid: its line
never ran at legacy) and `defaults_all5` (the NN winner flipped between runs; see the NN section). Both are kept,
superseded by `defaults_final`.

**Earlier runs, kept and superseded:**
- `defaults_lgb_probe` and `defaults_all`: the sidecar was not scrubbed and the NN could not find
  `train_single_trial.py`.
- `defaults_all2`: see the incident below.

## Incident during the work (my harness; fixed; disclosed)

**What was written where.** Run `defaults_all2` (2026-11-25 18:38:19 to 18:43:16). The test trees were built with
`models/` as a real directory whose entries were symlinks into this worktree. One of those entries was the directory
`models/reinforcement`. Step 5's `--output-dir <tree>/models/reinforcement` therefore wrote through the link into
**this worktree's** `models/reinforcement/`:
- **modified, tracked:** `models/reinforcement/best_model.txt` (`git diff --stat`: 2457 insertions, 2055 deletions);
- **created, ignored:** `best_model.cbm`, `best_model.json`, `best_model.meta.json`, `best_model.pth`. None existed
  when the worktree was made (`data/ch14_land/logs/status_before_w1.txt` has no `models/reinforcement` line).

**How it was caught.** The suite's own final gate `LAND.wrote_nothing_outside_out` (worktree
`git status --porcelain --ignored`, before against after) failed. It listed `M models/reinforcement/best_model.txt`.

**What was undone.** In this scratch worktree only:
- `git checkout -- models/reinforcement/best_model.txt` (back to `7f4505a`'s blob);
- `rm` of the four files the run had created.

`git status --porcelain --ignored -- models/` is now empty. That run's equality results are void, because the models
were written outside the snapshotted tree, and they are not used.

**Proof that nothing else changed:**
- **Main tree.** `git -C ~/distributed_prng_analysis status --porcelain --ignored | sha256sum` was `2a106e09…d011d7`
  before the first suite (`data/ch14_land/logs/main_tree_status_before.sha`) and after every suite through
  `defaults_all3`. (Main has since moved to `579dbb7` by the owner's landings.)
- **Links.** All **20,332** symlinks in `defaults_all2`'s trees point into `~/tfm_worktrees/ch14diag-land/`; **0**
  point anywhere else. The harness could reach no other tree.
- **Other worktrees** (`find -newermt` over the run's window, every worktree and the main tree, `.git` pruned):
  - `ll6-land`: 477 files, **all under its own `data/`** (that agent's runs; its `models/` is untouched);
  - `_review`: 1 file (`s1125_reg_head.txt`, the lead's);
  - main tree: 1 file (`logs/e2e_session/TASK_REG_FOLLOWUP_20261125.txt`, the lead's task file);
  - nothing else.

**The test change that stops it recurring.** `build_tree` no longer links any sub-directory of `models/` except
`wrappers/`. `models/reinforcement` and the `models/test_*` model folders are output folders and are never linked
(`tests/test_ch14_diag_land_defaults.py`, `build_tree`). Every run's outputs are therefore regular files inside its own
tree under `--out`. The `wrote_nothing_outside_out` gate stays as the backstop; it has passed on every run since
(`defaults_all3`, `_all4`, `_all5`, `_final`).

## NN non-reproducibility on CPU, at `7f4505a` itself (measured)

**Label.** ALREADY RECORDED in general, NEW as a measurement:
- "Step 5 training is not" deterministic: `ch14-learn:docs/drafts/ch14_v1/PLAN_CH14_LEARNING_LOOP.md:209-210`;
- the unseeded split: `docs/CHAPTER_6_AUDIT.md:140`;
- GPU non-determinism: `docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md:571`;
- seeds and run-to-run noise on block 0: `ch14-learn:READY_WHAT_MOVES_20261124.md`.

The measurement for the NN on CPU at the base revision, with the split seeded, is NEW. `train_single_trial.py` calls no
torch seed: `grep seed` has no hit.

**What was measured.** Two Step 5 CLI runs of the **base code** (`base`, `base2`), identical in every input, per suite
run. Each used `--model-type neural_net --trials 2 --k-folds 2`, the same fixture, and main's shim at split seed 1125.
The shim's first-permutation sha256 was equal in every run, so the splits were identical.
- **Equal:** the Optuna proposals (TPE seed 42), so the parameters of the 4 trial calls are identical, and the command
  lines of those 4 child calls are identical.
- **Different, every time:**
  - the trial **scores**. In `defaults_all3`: base −0.0236 / −0.0423, base2 −0.0145 / −0.0074. In `defaults_all5`:
    base −0.0538 / −0.0923, base2 −0.0639 / −0.0417.
  - the trained weights: max |Δw| over same-shape float tensors 0.56 between `base` and `base2`, against 0.44–0.50
    between `base` and each new run;
  - both diagnostics reports, the sidecar's metrics (`training_metrics/r2` −0.025 against −0.090) and the study's trial
    values.
- **Which trial wins is therefore random.** In `defaults_all3`, base2 chose the 4-layer trial and base the 3-layer one.
  In `defaults_all5`, base chose 3 layers, base2 4, new_default 3, new_legacy 4 and new_full 4. The final model's
  architecture, its sidecar `/hyperparameters`, and the final child call's `--params` follow the winner.
- **Files:** `data/ch14_land/logs/nn_child_argv_check.txt`, plus the run folders under `data/ch14_land/defaults_all{3,5}`.

**What it means for a "defaults unchanged" gate that covers the NN:**
- **No value-equality claim is possible on the Step 5 CLI for the NN.** Two runs of the same code differ as much as base
  and new do. A gate that compared values would fail at random. A gate that excluded the NN silently would be vacuous.
- **What the landing gate does claim for the NN,** in `tests/test_ch14_diag_land_defaults.py`:
  1. The report and sidecar **structure** (JSON key paths) equals base at default and at legacy. Only `/hyperparameters`
     is masked, because it records which trial won.
  2. **Every command line Step 5 gives the NN child** equals base at default and at legacy. The 4 trial calls are exact.
     The final call is exact apart from `--params`, which must be one of that run's trial proposals. Mutant S04 (one
     extra flag at legacy) is killed.
  3. The **health check** gives base's result on base's own report, and on a fixed synthetic NN report.
- **What covers the NN's values:** `tests/test_ch14_diag_wave1.py::test_nn_legacy_report_identical_to_base`. It trains
  the NN with base code and with new code in-process, with `torch.manual_seed(0)`, and compares the full legacy report
  (losses per round included), killed by M21. Wave 1 `test_cd03_nn_bit_identical_and_probe` and wave 2
  `test_cd23_cd07_nn_bit_identical` compare the NN `state_dict` across diagnostics off / legacy / full, seeded.
- **What is NOT covered:** a change that alters the NN's values only on the CLI path. That would mean something
  between the child's command line and the seeded trainer, for example an environment variable or the NPZ the parent
  writes, would not be seen. The NPZ is covered for its keys only, by `test_cd04_mpo_npz_handoff[legacy]`.

## Limits

1. **The NN is not reproducible run to run on CPU, even at `7f4505a`** (measured; see the NN section). On the CLI it is
   checked by structure and by its child command lines. Its values are checked in-process with torch seeded. A
   CLI-only value change between the child's command line and the trainer is not covered.
2. **`best_model.cbm` differs between two identical `7f4505a` runs.** Whether CatBoost's predictions differ was not
   measured.
3. **The defaults suite uses a synthetic fixture:** 240 rows, the 91 run H feature names, random values, 2 trials,
   2 folds. It shows equality, not behaviour on real data. On this fixture the tree families write no report at
   legacy, so their health result is `absent` on all three settings. The `fixed`-rules flip is shown on the NN only.
4. **`test_ll_feature_settings` ran 51 gates.** The commit message of `7cac105` records 53/53 for "LL". My
   [UNVERIFIED] reading is that the difference is the two optional `--runh-root` live gates, not run here because they
   read the e2e-map worktree.
5. **Git bookkeeping and two remaining writes:**
   - `git worktree add` wrote git's bookkeeping under `~/distributed_prng_analysis/.git/worktrees/ch14diag-land`, as the
     task required.
   - Git printed "Too many levels of symbolic links" warnings while reading the test trees' `.gitignore` links under
     `data/`. They are harmless and leave nothing behind.
   - The base-revision copies are untracked `.py` files under this worktree's `data/` (Gate 22's sensitivity applies to
     this worktree only).
6. **No GPU or real-data run.** The branch notes' Ask (one Step 5 `--diagnostics-level full` on run H) is still open.

## What it changes for pools, or for how hits against k/1000 are measured

**Nothing at defaults.** Step 5's training outputs (models, sidecars, Optuna trials) are content-equal to `7f4505a` for
lightgbm, xgboost and catboost, and structurally equal for the NN. The default health grade is unchanged, so WATCHER's
action after Step 5 is unchanged. Nothing touches Step 6 or the grader.

**When the owner flips `grading_rules: fixed`:** the grade WATCHER acts on changes. On this fixture's NN report it went
from critical/RETRY to ok/PROCEED. That changes which Step 5 runs are retried, or which families are later skipped, and
so, indirectly, which models vote. The owner has ruled that a poor grade never removes a family. `full` only adds report
content; training is equal at every level.

## Draft questions for a Beta review (BEFORE-COMMIT: this changes Step 5 and its trial worker)

- **Q1 Defaults.** Do `--diagnostics-level legacy` (and its absence) and `grading_rules` legacy (and its absence) leave
  Step 5's outputs and the health result exactly as at `7f4505a`? Is the evidence in `test_ch14_diag_land_defaults.py`
  (control-excluded files compared by structure, the health check on the same report) sufficient, given the NN's
  run-to-run non-determinism?
- **Q2 Merge.** Is the optimizer merge (main's LL-3 plus the branch's CD-04/05/06/07 hunks) correct where the two touch
  argparse and the NN subprocess flags (`meta_prediction_optimizer_anti_overfit.py`, around `:3144-3270` and `:2610-2650`)?
- **Q3 Gates.** Does each default-unchanged gate fail when it should? Specifically the precondition gate, the 10 added
  mutants, and the two strengthened gates (W34, W35). Can any gate still pass vacuously?
- **Q4 Isolation.** Does any test still read another worktree or write outside its `--out` / `--basetemp`? (The harness
  incident above is the reason to ask.)
- **Q5 The `fixed` path.** The fixed rules change WATCHER's retry/skip behaviour. Are the stale-report guard (CD-07),
  the skip expiry (CD-09) and the history prune (CD-10), all default off, safe to land dark, before the owner flips them?

Stopped.
