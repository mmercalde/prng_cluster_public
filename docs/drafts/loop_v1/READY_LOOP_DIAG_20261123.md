# READY — TASK_LOOP_DIAG: Chapter 14 diagnostics wave 1 built (CH14-DIAG, 2026-11-23)

Worktree `~/tfm_worktrees/ch14-diag`, branch `ch14-diag`, base `41ad509`. Nothing committed, pushed or landed.
No GPU used (every test and run sets `CUDA_VISIBLE_DEVICES=` and forces CPU devices). No pipeline step was run
on run H data. The re-grade only read run H's existing Step 5 reports, with sha256 checks before and after.
Plan: `docs/drafts/ch14_v1/PLAN_CH14_DIAGNOSTICS.md` (wave 1, lanes A–C).

## Prior record

**Searches run** (worktree at `41ad509`, using `/bin/grep -rlniE` so `.json` files and ignored files are included). Scope:
`docs/PROJECT_FILE_CATALOG.md`, `docs/TB_RULING_*`, `docs/TB_RULING_REQUEST_*`, `docs/PROPOSAL_*`, `docs/CHAPTER_*`,
`docs/*_AUDIT.md`, `docs/TODO_CONSOLIDATED.md`, `docs/BACKLOG.md`, `docs/LEADS.md`, `docs/exchange/`, plus
`~/tfm_worktrees/*/docs/drafts`.
Patterns: `diagnostics[-_]level`, `grading_rules`, `re-?grade`, `gradient_probe`,
`compare_models_summary_latest`, `classify_columns`.
- The only hits are `re-?grade` in `docs/CHAPTER_14_AUDIT.md:447` and `docs/CHAPTER_14_TRAINING_DIAGNOSTICS.md:1437`. Both are about
  stale re-grading of one report by the health check (audit C3), not about an offline re-grade tool.
- The other patterns have no hits.
- The full prior record for every item built here is the plan's `## Prior record`
  (`PLAN_CH14_DIAGNOSTICS.md:13-96`), re-verified at `41ad509`.

**Labels:**
- CD-01, CD-02, CD-04, CD-05 and CD-06 are **ALREADY RECORDED** as R-60, R-61, R-64, R-62/R-63 and R-65
  (`docs/CHAPTER_CORRECTIONS_REPORT_20261111.md` §2.11).
- CD-03 is **ALREADY RECORDED** as R-76.
- CD-00 and CD-13 are **NEW** (plan §A, §B).
- Nothing here **CHANGES A RULING**. The two switches that would change WATCHER behaviour are left at today's value, and flipping them is owner question 1.

**Not searched:** the host, the ser8 archives, the private repository, and project files that are not in the repository.

## Plain summary

**What was built** (all of plan wave 1, lanes A, B and C):
- a health-check fix for the two wrong report keys (best round and gradient spread);
- grading of the compare-models winner;
- a read-only offline re-grade;
- per-feature gradients for the neural net from a separate eval-mode pass;
- feature names carried into every report;
- reports for all three tree families, on both the trial-worker path and the final inline path;
- a column-type tag on every importance and gradient line.

**Defaults.** Every change sits behind a declared setting whose default is today's behaviour:
- `--diagnostics-level legacy|full` on Step 5 and on the trial worker;
- `training_diagnostics.grading_rules legacy|fixed` in `watcher_policies.json` (absent means `legacy`).

**Training is bit-identical with diagnostics off, on at `legacy`, and on at `full`.** The tests compare the neural-net
`state_dict` (full-batch and mini-batch), the saved model bytes for xgboost and lightgbm, and the predictions and metrics
for all four families, on both the worker path and the inline path. A mutant that runs the probe in train mode fails that test.

**Tests:**
- **36 pass** on CPU (2 min 35 s, peak RSS 725 MB).
- **22 of 22 mutants are killed.** Each mutant is shown to be applied, to have run (marker file), and to fail on its own tagged assertion (165 s with 4 parallel jobs).

**Re-grade results.** These are instrument readings, not TFM evidence. TFM is judged only by pool hits against `k/1000` on unseen draws.
- **Main-tree archive, 107 checks.** Recomputing with today's rules reproduces the recorded grade in **107 of 107**. All are critical.
  - With the fixed rules: 68 critical, 22 warning, 4 ok, and 14 absent (empty reports).
  - On the 64 real complete reports: **38 critical and 26 not**, the same as the audit's independent count (`docs/CHAPTER_14_AUDIT.md:410`). Of the 38, 25 are critical because the model's own report already said so, and 13 because of a real early-stop ratio.
  - Of the 93 checks that can be matched by time to a compare-models run, **77 graded the neural net while another family had won** (catboost 68). This is B6, seen in the data.
- **Run H Step 5, 2026-11-23** (`check_neural_net.json` `run_env.date` 10:50). Only the neural net wrote a report; the three tree families wrote none (that run used today's code, B3).
  - Today's grade is critical only because of the B1 artefact (ratio 0.00).
  - The fixed grade is **still critical, now for a real reason**: the best validation epoch is index 2 of 18 rounds (ratio 0.11 ≤ 0.15). The net peaked at its third epoch and then ran 15 patience epochs.
  - Counting rounds from 1 instead gives 3/18 = 0.17, which is **warning**. That choice is owner question 2.

**Not done:** see the TODO list. Seeing all four families graded together needs one Step 5 run at `--diagnostics-level full`, which needs a go (Ask, below).

## What was built, with anchors (line numbers are in the edited worktree files)

| item | what changed | anchors |
|---|---|---|
| level switch | `DIAGNOSTICS_LEVELS`, `validate_diagnostics_level`; base class takes `diagnostics_level`, `names_source`, `column_classes`; `full` adds fields only (`_add_full_report_fields`) | `training_diagnostics.py:89-103`, `:323` |
| CD-01 | fixed rules read `best_val_round` (falls back to `best_round`, used only by the health check's own self-test fixture); ratio = (index + `early_stop_round_base`) / `rounds_captured`; fewer than `early_stop_min_rounds` (default 2) is not graded; a report with no rounds grades `absent`, not `ok` | `training_health_check.py:491` `compute_fixed_metrics`, `:527`, `:596` |
| CD-02 reader | fixed rules read `model_specific.gradient_health.gradient_spread_ratio` (old key as fallback) | `training_health_check.py:491-500` |
| CD-02 producer | at `full`, the spread comes from the newest round that carries `feature_gradients`, not only the last round | `training_diagnostics.py:464`, `:757` |
| CD-03 | eval-mode probe: a `requires_grad` copy of the first `gradient_probe_rows` (default 512) training rows, `torch.autograd.grad(loss, x)` only; hooks skip the probe pass; the original mode is restored | `training_diagnostics.py:435-462`, on-round block; `train_single_trial.py:656-670` |
| CD-04 | Step 5 writes `feature_names` into the private trial NPZ (not the 22-array survivor contract) only at `full`; the worker reads it if present; otherwise `feature_names_source: positional` | `meta_prediction_optimizer_anti_overfit.py:2544`; `train_single_trial.py:964` |
| CD-05 worker | Booster path at `full`: curves (MSE) by staged prediction after training; lightgbm adds a `record_evaluation` observer callback, because `lgb.train` trims the Booster to its best iteration; catboost uses `get_evals_result()` (RMSE squared); importance by gain (catboost: PredictionValuesChange) over every column | `training_diagnostics.py:1210` `tree_curves`, `:1275`, `:1316`; `train_single_trial.py:112-146`, `:247` |
| CD-05 inline | final inline sklearn-API tree model: report built after training (fit untouched) at `full`; writes `{model}_diagnostics.json` plus the canonical copy | `meta_prediction_optimizer_anti_overfit.py:2750`, call sites `:1848`, `:2826` |
| CD-06 writer | at `full`, compare-models also writes `diagnostics_outputs/compare_models_summary_latest.json` (the per-run file is kept) and passes the level to each child | `meta_prediction_optimizer_anti_overfit.py:239-243`, `:334-339` |
| CD-06 reader | fixed rules: pointer → `winner_best_effort.model_type` (a dict) → that family's own report; every family graded into `per_family`; `grade_scope winner|all_families` (default `winner`); a winner with no report grades `absent` | `training_health_check.py:604`, `:620` |
| CD-13 | `classify_columns`: `constant` / `run` / `trial` / `local` computed from the training matrix; `attribution_share_by_class`; every top-10 line tagged `column_class` | `training_diagnostics.py:1110`, `:1176` |
| CD-00 | `python3 training_health_check.py --regrade PATH... --out X.json --policies P`: pure `grade_report` (no registry, no archive); legacy, fixed and fixed-base-1 side by side; sha256 of every input before and after, failing if any changed; refuses an existing `--out` or one inside an input | `training_health_check.py:573` `grade_report`, `:1054` `regrade`, `:1230` |
| legacy path | `_check_training_health_impl` runs the original body when the rules are `legacy`; grading split into the side-effect-free `_grade_single` / `_grade_multi` | `training_health_check.py:163`, `:265`, `:286` |

**Hunks in the shared files, for the merge** (`git diff -U0` headers, worktree vs `41ad509`):
- `meta_prediction_optimizer_anti_overfit.py`, 10 hunks:
  - `@@ -238,0 +239,5` child level flag
  - `@@ -328,0 +334,6` pointer
  - `@@ -1836,0 +1848` inline call (Optuna final)
  - `@@ -2532,3 +2544,10` NPZ names
  - `@@ -2588,0 +2608,4` NN subprocess flags
  - `@@ -2721,0 +2745,39` `_ch14_full` / `_ch14_emit_inline_tree_report`
  - `@@ -2763,0 +2826` inline call (single-shot)
  - `@@ -3081,0 +3145,9` argparse
  - `@@ -3123,0 +3196,4` compare args
  - `@@ -3184,0 +3261,2` optimizer attrs
- `train_single_trial.py`, 18 hunks:
  - tree emit helpers `@@ -112`, `@@ -115`;
  - tree signatures and emit calls `@@ -180`, `@@ -262`, `@@ -277`, `@@ -356`, `@@ -371`, `@@ -435`;
  - lightgbm callback `@@ -212`, `@@ -215`, `@@ -235`, `@@ -237`;
  - NN signature `@@ -453`, NN attach `@@ -614`;
  - main: argparse `@@ -875`, NPZ names `@@ -896`, options and calls `@@ -936…-952`.
  - **Overlap with `selfplay-nn`**: both edit the `train_neural_net` signature line (`@@ -453`). Its other hunks
    (the normalisation block and the checkpoint keys) do not touch mine. The merge is a two-kwarg union.

## Tests

- **Files:** `tests/test_ch14_diag_wave1.py` (36 tests), `tests/ch14_cpu_patch.py` (forces the CPU device for tree libraries; imports no code under test), `tests/ch14_wave1_mutants.py` (22 mutants).
- **Run:**
  - Tests: `CUDA_VISIBLE_DEVICES= python3 -m pytest -q tests/test_ch14_diag_wave1.py --basetemp=data/ch14/pytest_tmp -p no:cacheprovider` → **36 passed in 153 s; wall 155 s; peak RSS 724,676 kB** (`data/ch14/logs/pytest_final.log`, sha256 `86244d1e…a0f8`).
  - Mutants: `python3 tests/ch14_wave1_mutants.py --out data/ch14/logs/mutants_final.json --jobs 4` → **22/22 killed, wall 165 s, peak RSS 558,272 kB** (`mutants_final.json`, sha256 `44891517…e9`; per-mutant logs and marker files are in `data/ch14/logs/mutant_logs/`).
- **Multi-core:** the mutant runner uses 4 parallel pytest processes, and the xgboost, lightgbm and catboost fits use all cores. The pytest suite itself is one process.
- **Gates:**
  - "Default unchanged" against `41ad509`, by `git show` copies: the health-check result plus skip registry plus archive for 7 cases (late best, early best, old spread key, multi-model, absent, unreadable, self-test fixture), each checked twice so the registry path RETRY→RETRY runs; the legacy NN and tree worker reports, byte-for-byte apart from `generated_at`; the legacy NPZ keys; and no inline report at `legacy`.
  - **Bit-identity:** listed in the plain summary above.
- **Mutant set:**
  - one per item: key, spread key, producer last round, probe skipped, names dropped (worker and Step 5), tree full branch off, xgboost curve one point only, lightgbm callback removed, inline never emits, old pointer filename, winner dict ignored, pointer not written, child flag not threaded, classifier off, re-grade fixed column using legacy;
  - three that check the gates themselves: a re-grade that writes an input and skips the guard, legacy rules silently "fixed", and legacy reports gaining `full` fields;
  - two that change training (probe in train mode, a lightgbm callback that changes the learning rate), each caught by the bit-identity assertion.
- **One invalid mutant run.** An earlier mutant run failed because the pytest `--basetemp` parent folder was missing. The runner scored those as survived (fail-closed). That log was replaced by the final run.
- **Settled UNVERIFIED #5:** the xgboost 3.0.4 and lightgbm 4.6.0 Booster objects have no `feature_importances_`, so the worker path never recorded tree importances. The test runs this on CPU.
- **Measured fact, not changed:** the worker's xgboost `Booster.predict` uses every boosted round, including the 20 after the best iteration (`default==all True`, `default==best False`, CPU probe this session; `train_single_trial.py:334-335` at `41ad509`). Whether that matters for hit rate is a question for the grader, not for this item.

## Re-grade outputs (CD-00), read-only

| input | output | sha256 |
|---|---|---|
| `~/distributed_prng_analysis/diagnostics_outputs` (223 files hashed, 0 changed) | `data/ch14/regrade_main_20261124T015255Z.json` | `8b94feb1…15d6` |
| `~/tfm_worktrees/e2e-map/data/e2e_runH/step45` (40 files hashed, 0 changed) | `data/ch14/regrade_runH_20261124T015255Z.json` | `61651edf…c3a` |

Grades below are the recorded or today's grade → the fixed grade. Bounds come from the main tree's `watcher_policies.json` (identical to the worktree's).

| group (main tree) | n | today | fixed | fixed, base 1 |
|---|---|---|---|---|
| producer schema 1.1.0, complete, neural_net | 60 | 60 critical | 34 critical · 22 warning · 4 ok | 28 · 28 · 4 |
| producer schema 1.1.0, complete, catboost | 4 | 4 critical | 4 critical (producer critical + early stop + overfit) | 4 |
| schema 1.1.0, `partial`, no rounds | 14 | 14 critical | 14 absent | 14 absent |
| no schema, `diagnosis` + `metrics` only (not producer shape; origin **UNVERIFIED**) | 27 | 27 critical | 27 critical (producer severity) | 27 |
| health-check self-test fixture (`best_round` 15/100) | 2 | 2 critical | 2 critical | 2 |
| live `neural_net_diagnostics.json` (2026-03-06) | 1 | critical | critical | critical |

**Run H step45:**
- `neural_net/diagnostics_outputs/{training,neural_net}_diagnostics.json` are identical (same sha256 `4420e9e7…`): today critical, fixed critical (early stop 2/18), base 1 warning.
- The other 4 reports are e2e-map's own `selftest/nnA`, `nnB` fixtures, not run H data: critical (overfit 7.12 and 3.99, and the producer said critical).
- lightgbm, xgboost and catboost: **no report exists** (B3; today's code).

## TODO (shortcuts, each with its anchor)

- **TODO-CH14-1.** WATCHER cannot yet choose the level or the rules. `--diagnostics-level` needs a Step 5 manifest entry
  (`agent_manifests/reinforcement.json`, hand-landed, Beta before-commit), and `grading_rules: fixed` needs a line in
  `watcher_policies.json`. Both are left at today's behaviour; owner questions 1 and 3.
- **TODO-CH14-2.** No stale check yet: the pointer and the family reports carry no run id (CD-07, wave 2,
  `training_health_check.py:620` docstring).
- **TODO-CH14-3.** Reports are still written to the flat `diagnostics_outputs/`, not to a per-run folder `{run_id}/` (CD-07 / CD-20).
- **TODO-CH14-4.** CD-13 on the Step 5 path reports `trial_class: unavailable` (no trial id in the X rows,
  `meta_prediction_optimizer_anti_overfit.py:621-691`, rows built at `:662`, `:689`), and no run-level name set is declared, so the `run` class is
  never assigned there. Both hooks exist (`groups`, `run_level_names`, `training_diagnostics.py:1110`).
- **TODO-CH14-5.** The tree importance kinds differ: xgboost and lightgbm report `gain`, catboost `prediction_values_change`. Each report records its kind.
- **Not in this task:** CD-17 (lane D in the plan's wave 1), and all of waves 2 and 3.

## UNVERIFIED

- The origin of the 27 archived "no schema" checks (issues such as `gradient_spread_extreme`, `metrics.dead_neuron_ratio`, archived 2026-02-15): they look like injected test reports. Not traced.
- The time match between archived checks and compare-models runs (run start ≤ check time, ≤ 6 h, setting `--match-window-hours`) is a heuristic. The archive does not record which run a check followed.
- The full-level reports have been seen only on synthetic CPU fixtures, not on a real Step 5 (that needs the go below).
- Wall times for the full level at production scale (staged prediction is one predict per round per split) are not measured.

## Ask (needs the lead's go, and the owner's for run H data)

One Step 5 `--compare-models --enable-diagnostics --diagnostics-level full` on run H's Step 3 output, in a scratch
copy, on the VM101 card, then `--regrade` over its `diagnostics_outputs/`. That is the first time all four families
are graded together, with named, class-tagged importance and NN gradients. No GPU job has been started.

## Owner questions (one line each)

1. Flip `watcher_policies.json` `training_diagnostics.grading_rules` to `fixed`? On the archive, 40 of 108 checks stop being critical, so WATCHER retries and skips less.
2. Should `early_stop_round_base` be 0 (best index / rounds, as the policy text says) or 1 (a 1-based round count)? It decides critical versus warning for run H's NN.
3. Should WATCHER's Step 5 run at `--diagnostics-level full` by default once a manifest entry exists (hand-landed by you)?
