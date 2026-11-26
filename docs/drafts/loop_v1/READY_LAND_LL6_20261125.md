# READY · LL-6 landing prepared: one inference bundle for Step 6 and Chapter 13 (2026-11-25)

## After Beta R1 (lead TASK_LL6_BETA_Q5_FIXES_20261125, 19:40)

- **Thread:**
  - `X-LL6-BUNDLE` R1: request `a6182534…`, review `docs/exchange/from_beta/X-LL6-BUNDLE_R1_BETA_REVIEW.md` `a3ad5d1f…`, both uncommitted in the main tree; relay verdict `REVIEW RECEIVED`;
  - my summary: `docs/drafts/loop_v1/BETA_R1_LL6_SUMMARY_20261125.md`;
  - the R2 reply is drafted here, **not sent**: `docs/drafts/loop_v1/X-LL6-BUNDLE_R2_ALPHA_REPLY.md`. Whether R2 is sent, or the owner lands without a second round, is the owner's call.
- **Main moved again:** it is now `ed2d22d` (20:15). That commit holds the R1 exchange files of three threads, including this one; docs only, no LL-6 path (`git diff --stat 579dbb7 ed2d22d`). It came after this step's proof, whose ref digests were taken on `579dbb7`.
- **Status:** nothing committed anywhere; no relay run in this step; no card. All seven Q5 fixes are in the test files. **The three shipped files are byte-identical to what Beta reviewed** (sha256 checked at the end, table below).

### Plain summary

1. **Beta's positions:** Q1, Q2 and Q4 `NO OBJECTION`; Q3 `SUGGESTION`; **Q5 `OBJECTION · LIVE`**. The lead accepted Q5 as real; the label is not contested.
2. **All seven proposed test fixes are applied,** in `tests/test_ll6_inference_bundle.py` and `tests/ll6_step6_runner.py`.
3. **A further fault was found while fixing Q5, and fixed in the runner.** The Step 6 CLI log comparison compared **0 log lines** in every run, in R1 too, because an import made by Step 6 leaves the root logger at WARNING. R1's "all log lines equal" was therefore vacuous. It is now 38-42 lines per family, plus stdout and stderr, and a mutant proves the comparison is live.
4. **Fresh run:** LL-6 test **14/14 gates, 13/13 mutants**. Every mutant's target gates are green unmutated and red mutated.
5. **The same landed suites pass at their earlier counts:** walk-forward selftest 108/108; git refs, main tree status, Optuna listing and `/tmp/matplotlib-*` unchanged.
6. **The three shipped files are unchanged:** `586a3375…`, `4f6a26bf…`, `40e1fcb3…`.

### Beta's positions and what was done

| Q | Beta | What was done |
|---|---|---|
| Q1 Default behaviour | NO OBJECTION | Recorded. **One correction to my R1 evidence:** the CLI log comparison compared no lines (item 3 above). It now compares 38-42 log lines per family plus stdio (below). |
| Q2 Canonical loader and A-NORM | NO OBJECTION | Recorded. **Beta's evidence, labelled as Beta's** (review §Q2): on all 12,156 development block-0 Step 3 rows with the four archived models, this landing's Step 6 equals the archived CPU scores exactly; checkpoint hashes equal their frozen copies; Chapter 13 canonical input transform plus attribution-object forward against Step 6: NN `7.46e-9`, xgboost, lightgbm and catboost `0` (atol 1e-5). Beta also captured Chapter 13's 20 real NN backend forwards in eval mode: ≤ `9.38e-9` against Step 6. Not measured: an end-to-end Chapter 13 draw-to-artifact cycle; hit rate. |
| Q3 NN mode, tier comparison | SUGGESTION | **Adopted as a recorded follow-up for LL-7/LL-8,** not in LL-6. Put the tier fix with LL-7 or LL-8 before tier output is relied on: one validated vector and scaler feed both the attribution and the tier path, tested with named rows in a regime-shift case that checks the tier output. Eval mode at bundle load is optional (both current consumers call `eval()`). |
| Q4 Scaler source | NO OBJECTION | Recorded. **Beta's evidence:** relocated-directory gap `4.55` (`resolved`), `0` (`as_written`) on Beta's fixture; mine `1.89`, now `2.19` on this run's fixture. Added to the limits: A-NORM fails by construction for a relocated NN directory, because Step 6 then has no scaler; check parity on the actual inference paths before reading such an attribution. |
| Q5 Tests after landing | OBJECTION · LIVE | **Fixed** (below). |
| Finding 1 (prior record) | — | **Added** to §Prior record (below). |
| Finding 2 | THEORETICAL | **Added to the limits** (below). Not added to `DEFERRED.md` (lead: do not edit it). |

### Q5: what changed in the tests

| # | Fix | Where (ll6-land lines) | Proof |
|---|---|---|---|
| 1 | Temp and cache roots under `--out` (`TMPDIR`, `XDG_CACHE_HOME`, `MPLCONFIGDIR`, `CUDA_CACHE_PATH`), set before numpy or any library import, inherited by every child: Step 5 training, Step 3 fixture runner, Step 6 runner, mutant runs | `tests/test_ll6_inference_bundle.py:91-100` `redirect_env_roots`; called first in `main()` | the summary records the four roots under `ll6_Q2/selftest/_env/`; `/tmp/matplotlib-*` count 0 before and after the whole proof |
| 2 | No-write gate on the exact boundary: git status of the tree outside `data/` (tracked, untracked, ignored); a walk of every path under `data/` with only the selected `--boundary` pruned; new `/tmp/matplotlib-*`; `~/.cache/matplotlib` and `~/.config/matplotlib`. `--boundary` (default `--out`) must contain `--out` and lie under `data/`; mutant runs get the parent's boundary | `boundary_snapshot` `:103`, `boundary_diff` `:131`; gate `LL6.no_write_outside_out` | **negative control** `LL6.no_write_gate_catches_sibling_write`: a real sentinel file under `data/`, outside the boundary, is caught, and after removal the snapshot is clean; a synthetic new `/tmp/matplotlib-*` entry is caught. In development the gate also caught my own `tee` log written beside `--out` (`data/ll6_prep/q5_dev1.log`), which is why each proof run now has its own run directory as `--boundary` |
| 3 | Step 6 CLI log control. A log line counts as volatile only if it differs between the two baseline runs **and** looks like a clock or run id (`VOLATILE_LINE`); otherwise new = base1 = base2. The compared line count must be > 0. stdout and stderr are compared too, with one declared normalisation: torch's `FutureWarning` for `torch.load` names its call site, which LL-6 moved from `prediction_generator.py` to `models/inference_bundle.py:81` (same call, same arguments) | `norm_stdio`, `TORCH_LOAD_WARNING`; gate `LL6.step6_cli_default_output_identical_to_base`; runner `tests/ll6_step6_runner.py` re-installs its capture after the import (root level was 30 = WARNING; `logging_after_import` recorded) | **42 / 38 / 38 / 38 log lines** compared (neural_net, xgboost, lightgbm, catboost; volatile 0), equal on all three runs of each family; stdout and stderr equal too (5 lines for neural_net, which carries the moved `torch.load` warning, 1 for each tree model); mutant `scaler_log_changed` now also turns this gate red (non-vacuity of the log check) |
| 4 | The real Chapter 13 handoff. `run_cycle()` → `_load_best_model_if_available()` (canonical) → the call at `chapter_13_orchestrator.py:355-366` → root cause. Stubbed: diagnostics, a stand-in `database_system` that refuses to construct, the regression check, the predictions on disk. The cycle is stopped by a `BaseException` right after root cause, so no trigger, LLM or cycle log runs; cwd under `--out` | gate `LL6.c13_run_cycle_hands_bundle_to_root_cause` | bundle handed over (checkpoint hash equal), 20 attribution inputs equal `bundle.transform(bundle.vector(features))`, 18/2 attributed. **Mutant `handoff_bundle_dropped`** (`bundle=model_info.get('bundle')` → `bundle=None`, Beta's mutant) turns it red |
| 5 | Wrong-checkpoint mutant, carried over from W2: a second NN fixture of the same family and width (`models_neural_net_alt`, another seed) replaces the model inside the bundle | mutant `wrong_checkpoint_same_family` (`load_model_from_sidecar(models_dir=…LL6_ALT_NN_DIR…)` for the NN only) | A-NORM red; marker hit |
| 6 | Provenance: the summary records the `--baseline-rev` actually used, beside the default, plus the boundary and the env roots | `finish()` | `ll6_Q2/selftest/selftest_summary.json`: `baseline_rev` `7f4505aaccebdd284393eb3565e75d09d1c0f3b4` |
| 7 | Mutant credit: a mutant counts only if every target gate is **green in the unmutated run** and red under the mutant (and the marker hit where declared). Both states are printed | `run_mutants` | 13/13 credited with `unmutated` all `True` |

### Results after the fixes (VM 101, CPU, one at a time, `taskset -c 0-2 nice -n 10`, `python3 -u -B`; runner `data/ll6_prep/run_Q.sh`)

| Suite | Result | log sha256 |
|---|---|---|
| `tests/test_ll6_inference_bundle.py`, fresh fixtures, final code (`data/ll6_Q2/`) | **14/14 gates, 13/13 mutants**, 222 s (fixtures 34 s), RSS 637 MB, largest child 635 MB | `99d70a46…` |
| `tests/test_ll6_inference_bundle.py`, the first fresh run before the log-capture fault was found (`data/ll6_Q/`) | 14/14, 13/13, with 0 log lines compared (vacuous log check) | `9d5a2d5a…` |
| `tests/test_ll_feature_settings.py`, default / `--runh-root` | 51/51, 53/53; mutants 15/15 | `b53670c9…`, `63d7ac41…` |
| `tests/test_e2e_vote_harness.py` | 15/15, 17/17 | `15f69562…` |
| `tests/test_e2e_vote_ll5.py`, `_ordering`, `test_e2e_miss_report.py` | 3/3 3/3, 6/6 3/3, 7/7 5/5 | `a0addec0…`, `9c9d0e65…`, `3a617e39…` |
| `tests/test_step1_screen.py` | GATES 16/16, MUTANTS 15/15, SENTINEL PASS | `92612700…` |
| `scripts/diagnostics/e2e_walk_forward_selftest.py` (started 19:53 after `pgrep -af beta_relay` was empty; `data/ll6_prep/Q_wait.txt`) | **108/108** | `4c309b3a…` |

- **The landed suites ran once,** against the final shipped files. The last runner and test change affects only the LL-6 test, which was re-run fresh afterwards.
- **Commits (owner ruling 19:26):**
  - no LL-6 test and no suite run here commits; `test_step1_screen.py` only reads a pinned commit, and no other suite in the set calls `git commit`;
  - the git ref digest of the repository (`git for-each-ref`, sha256 `2534060d…`) and both HEADs (main `579dbb7`, tree `7f4505a`) are identical before and after the proof (`data/ll6_prep/Q_refs.txt`, and the Q2 line in `Q_summary.txt`).
- **Main tree:** `git status` identical before and after the walk-forward selftest; Optuna listing unchanged; `/tmp/matplotlib-*` none before or after.

### Landing list, final sha256

| # | Path | sha256 | changed after Beta read it |
|---|---|---|---|
| 1 | `models/inference_bundle.py` (N, **force-add**, `.gitignore:162`) | `586a33752363ef5821bbc31b991de1355c5baf3c0b78c8a263f232a9fb904be6` | no |
| 2 | `prediction_generator.py` (M) | `4f6a26bff13d854d928afd623e1a912fb94d93fa8e98cd0dc0793e3e522bd808` | no |
| 3 | `chapter_13_orchestrator.py` (M) | `40e1fcb326801d6bb3035d981f5ad48f3bf07815bfa8a1b6dba52a6ef5f0906b` | no |
| 4 | `tests/test_ll6_inference_bundle.py` (N) | `89de4a5a6117a4db730ba8dd9c51dee82aa9dfd17f3ad88d7c2c286952526109` | **yes** (was `ed703f2f…`) |
| 5 | `tests/ll6_step6_runner.py` (N) | `d69c7a373b99025b0fb74bb6daf6ed659f1b194ba12dc0565c9a7a2b27052524` | **yes** (was `77a910d2…`) |

- Plus this note, and the drafts `BETA_R1_LL6_SUMMARY_20261125.md`, `POSTLAND_CHECK_579dbb7_20261125.md` and `X-LL6-BUNDLE_R2_ALPHA_REPLY.md` in `docs/drafts/loop_v1/`.
- The R1 copies of rows 4 and 5 are kept in `data/ll6_prep/pre_q5_*.py`.
- **Force-add:** row 1 only.

### Limits added after Beta R1

- **THEORETICAL (Beta Finding 2):** a precondition before `canonical` is used on real artifacts.
  - The canonical adapter accepts a positional vector by width alone (`chapter_13_orchestrator.py:89-99`).
  - The root-cause analysis does not bind a ranked-prediction record to the bundle's schema or checkpoint.
  - With a same-width reordered vector, or an artifact from another checkpoint, the report could describe the wrong inputs.
  - The system map §4.5 contract must be enforced at the canonical boundary first; the missing trigger inputs and artifact producer are Chapter 13 audit B15. It holds nothing for this landing, because `canonical` is opt-in and observe-only.
- **Q3 follow-up (LL-7/LL-8):** the tier comparison through the same validated vector and scaler, tested with named rows.
- **Q4:** A-NORM fails by construction for a relocated NN directory, because Step 6 has no scaler there. Check parity on the actual inference paths before reading such an attribution.
- **The R1 Step 6 log check was vacuous** (0 lines), as stated above. The R1 output-key and prediction comparisons were not affected.

---

## Before Beta R1 (the state Beta reviewed)

- **Task:** `~/distributed_prng_analysis/logs/e2e_session/TASK_LL6_LAND_PREP_20261125.txt` (lead, 18:22).
- **Branch survey:** `~/tfm_worktrees/e2e-map/docs/drafts/loop_v1/SURVEY_CH14_LEARN_20261125.md` §2 U1 and §4.
- **Status:**
  - nothing committed anywhere, and no scratch commit; nothing pushed or landed;
  - no Beta relay run; no card, no rig;
  - nothing written in `~/distributed_prng_analysis`.
- **Tree:** `~/tfm_worktrees/ll6-land`, detached at `7f4505aaccebdd284393eb3565e75d09d1c0f3b4`. The five landing files are working-tree changes. Evidence is in `data/ll6_prep/` and `data/ll6_P_selftest/`.
- **Main moved during the work:** it is now `1c95157` (local, `origin` and `public`). That commit changes `CLAUDE.md` only (`git diff --stat 7f4505a 1c95157`), so none of the LL-6 files and none of the baselines change.

## Prior record

**Searches run** (main, current HEAD):
1. `git grep -lE "<pattern>" HEAD --` over `docs/PROJECT_FILE_CATALOG.md`, `docs/TB_RULING_*`, `docs/TB_RULING_REQUEST_*`, `docs/PROPOSAL_*`, `docs/CHAPTER_*`, `docs/*_AUDIT.md`, `docs/TODO_CONSOLIDATED.md`, `docs/BACKLOG.md`, `docs/LEADS.md`, `docs/exchange/`, the system map and `docs/drafts/`. Patterns:
   - `inference_bundle|inference bundle`: the system map, two landing READY notes, and the draft registry patch `docs/drafts/loop_v1/switch_patches_20261123/parameter_registry.py.patch(ed)`;
   - `c13[-_]model[-_]loader`: `docs/drafts/loop_v1/SWITCH_CHECK_20261123.md` and that same draft registry patch;
   - `A-NORM`: Chapter 13 audit, Chapter 14 audit, Chapter 14 training diagnostics, the system map, TODO, `MAP_E2E_THIN_PATH_20261122.md`;
   - `Blocker 3`: Chapter 13 audit, Chapter 14 audit, Chapter 14 training diagnostics, KPI proposal and ruling (another subject), the system map.
2. `grep -lE "LL-6|inference bundle|c13-model-loader|A-NORM" logs/e2e_session/*.txt` found 6 files: `TASK_LOOP_ADDENDUM_3.txt`, `TASK_LEARN_WHY_BLOCK0_20261124.txt`, `TASK_AFTER_WF_VOTE_LANDING_20261125.txt`, `TASK_CH14_LEARN_SURVEY_20261125.txt`, `TASK_CH14_DIAG_LAND_PREP_20261125.txt`, this task.
3. The ch14-learn survey's own prior record (`SURVEY_CH14_LEARN_20261125.md`).

**Documents that decide or record the subject:**
- `docs/TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md`:
  - `:95-111`: Blocker 3, gate A-NORM "(release-blocking)", and Beta's ruling "One-line Ch13 feature-name patch alone — Not sufficient; do not land in isolation";
  - `:256`: the activation list ("use the canonical sidecar/model loader in Ch13; restore the exact NN scaler; validate A-NORM").
- `docs/CHAPTER_13_AUDIT.md:153-155` (Blockers 2-4) and `:212-214` (what must change).
- `docs/TODO_CONSOLIDATED.md:48`, `:120`: P0-C, "canonical model loader in Ch13, NN scaler, A-NORM parity gate".
- `logs/e2e_session/TASK_LOOP_ADDENDUM_3.txt:14`: the lead ordered wave 2, "LL-6 (Chapter 13 loads the model the way Step 6 does, A-NORM gate, all four families, CPU)".
- `logs/e2e_session/OWNER_RULING_switch_registration_order_20261125.txt`: the order is registry repair, then switch registrations, then manifests. It covers the registration of `--c13-model-loader`, which is not part of this landing.
- The build record on the branch: `~/tfm_worktrees/ch14-learn/docs/drafts/loop_v1/READY_LOOP_LEARN_W2_20261123.md`, not on main.
- **Added after Beta R1, Finding 1:**
  - `logs/e2e_session/OWNER_DIRECTION_learning_loop_team_20261123.txt:2-6`: the owner's learning-loop direction, which authorises the Chapter 13/14 why-work and makes the next graded block the loop's test;
  - `logs/e2e_session/TASK_LEARN_WHY_BLOCK0_20261124.txt:2-5` and `~/tfm_worktrees/ch14-learn/docs/drafts/loop_v1/READY_WHY_BLOCK0_20261124.md:120-129`: the earlier, development-only, real-model A-NORM on the branch (block 0, four families; NN 1.5e-8, trees 0);
  - `docs/TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md:113` §4.5, "The positional-alignment prohibition", material to Q2 and to Beta's Finding 2.

**Labels:**
- `ALREADY RECORDED`: the LL-6 design and its rulings (above); the relocated-directory scaler difference (W2 note, gate `LL6.relocated_dir_step6_drops_scaler`).
- `NEW`:
  - this landing's test file and runner, its pins and mutants;
  - the module the bundle hands over is in training mode (§5);
  - with the canonical loader and name-keyed predictions, the regime-shift tier comparison fails non-fatally (§5).
- `CHANGES A RULING`: none.

**Not searched:** ser8 archives, the private repository, the rigs, and project files outside the repository.

## Plain summary

1. **LL-6 is ready to land: 5 files, and nothing changes at default.** Only `models/inference_bundle.py` needs a force-add.
2. **Step 6's default output equals `7f4505a`.** Four families went through the real Step 6 entry point: 35 of 35 output keys and all log lines are equal; only two timestamps differ, as in a control repeat of the baseline. *(Corrected after Beta R1: the "log lines equal" part compared 0 lines at this stage; see "After Beta R1".)*
3. **Chapter 13's default loader (`legacy`) equals `7f4505a`:** loader result, root-cause result and the default a caller gets by passing nothing.
4. **Non-vacuity.** The canonical loader gives a different load than the baseline for all four families. The three mutants that break the scaler change Step 6's output.
5. **A-NORM passes for all four families.** For every fixture survivor, Step 6's prediction matches the attribution path's own forward pass: NN 3.7e-9, the three tree models 0 (declared atol 1e-5).
6. **New test `tests/test_ll6_inference_bundle.py`: 12/12 gates, 11/11 mutants killed.** Its baseline is pinned to `7f4505a` by full sha. It reads no other worktree and writes nothing outside `--out`.
7. **The landed suites pass at their earlier counts.** LL 51/51 and 53/53 (15/15 mutants); vote harness 15/15 and 17/17; walk-forward selftest 108/108; the four smaller suites in §3 equal too.
8. **Limits (§5).** NN only on a 200-row CPU fixture. Two canonical-path gaps are measured but not fixed: the module comes back in training mode, and the tier comparison is skipped. No hit-rate evidence.

## 1. What LL-6 is, and what changed for landing

- **What it does.** `models/inference_bundle.py` loads a Step 5 model the way Step 6 does, as one explicit bundle (model, type, sidecar names, schema hash, feature count, NN scaler, checkpoint hash).
  - Step 6's scaler reader moves into it unchanged, and `PredictionGenerator._load_nn_scaler` calls it (`prediction_generator.py:389-406`).
  - Chapter 13 gains `--c13-model-loader {legacy,canonical}`, default `legacy`. The `legacy` path is the old body, untouched (`chapter_13_orchestrator.py:586-589`). `canonical` uses the bundle (`_load_model_canonical`, `:674-689`) and feeds root cause scaled, name-keyed inputs (`_c13_model_input`, `:89-99`).
- **`prediction_generator.py` = main's file plus the LL-6 hunk only.** It was built by `git merge-file` (base `41ad509`, ours `7f4505a`, theirs `7368f34`), which merged cleanly. `diff` against main shows only the two adjacent regions inside `_load_nn_scaler` (`data/ll6_prep/pg_main_vs_landing.diff`). The result is byte-equal to the branch copy, because main's LL-1 hunks equal the branch's.
- **`chapter_13_orchestrator.py`** is the branch copy. Main's file has not changed since the merge base (`ae049c9e…` at both `41ad509` and `7f4505a`), so the branch copy is that file plus LL-6.
- **`models/inference_bundle.py`:**
  - copied from the only copy, unchanged (`586a3375…`);
  - first copied, never moved, to `~/tfm_worktrees/_records/inference_bundle_20261125/` with a `SHA256SUMS` (`sha256sum -c` OK);
  - the ch14-learn original is still in place.
- **Tests: new, written for this landing.** ch14-learn's `tests/test_ll_wave2.py` is not landed. It mixes LL-6/7/8, reads `HEAD`, and writes `diagnostics_outputs/history` into the worktree. The new files:
  - `tests/test_ll6_inference_bundle.py` takes the LL-6 gates from it.
  - **A-NORM uses the bundle's attribution object and an independent forward pass,** not LL-7's signed-attribution functions.
  - **Every "unchanged" gate compares with `git show 7f4505aaccebdd284393eb3565e75d09d1c0f3b4:<file>`.** `--baseline-rev` must be a full 40-hex sha; `HEAD` is refused. A precondition gate requires the baseline to lack LL-6 and the tree under test to have it.
  - **Step 6's default output goes through the real `main()`,** via the new `tests/ll6_step6_runner.py`, with a repeat of the baseline as the control.
  - **Chapter 13's root-cause calls run with their cwd under `--out`.**
  - **Mutants are source mutations:** one exact replacement in a copy of the file, in a code dir placed first on `sys.path`.
  - **Fixtures are synthetic:** `make_fixture` and the Step 3 worker (both on main), and four models trained by the real Step 5 CLI (`--trials 1`, single-shot, storage inside `--out`).

## 2. Landing list (against main `7f4505a` = `1c95157` for these paths)

| # | Path | N/M | sha256 | on main |
|---|---|---|---|---|
| 1 | `models/inference_bundle.py` | N, **force-add** (`.gitignore:162` `models/`) | `586a33752363ef5821bbc31b991de1355c5baf3c0b78c8a263f232a9fb904be6` | — |
| 2 | `prediction_generator.py` | **M** (pipeline, Step 6) | `4f6a26bff13d854d928afd623e1a912fb94d93fa8e98cd0dc0793e3e522bd808` | `3cc11eb1b3add4d4d3139bd5e957ff07f37bc806afe03aaffe88e3cf0c7c8fa2` |
| 3 | `chapter_13_orchestrator.py` | **M** (Chapter 13) | `40e1fcb326801d6bb3035d981f5ad48f3bf07815bfa8a1b6dba52a6ef5f0906b` | `ae049c9ec2705245cd02ea3c94f1cbd510c8ea13bf0133828f9bf5902fbd5cd7` |
| 4 | `tests/test_ll6_inference_bundle.py` | N | `ed703f2f0c2fefe3b58732a72b33eb8509d313a5d11775b16064f6b19610c8ec` | — |
| 5 | `tests/ll6_step6_runner.py` | N | `77a910d2f13c1b865489b7bb8611fc6f1b7e44792399cc188a2e85d56b0c5a06` | — |

- Plus this note, `docs/drafts/loop_v1/READY_LAND_LL6_20261125.md` (N).
- **The owner force-adds:** row 1 only, `git add -f models/inference_bundle.py`. Check afterwards with `git ls-files models/inference_bundle.py`. `git check-ignore -v` reports no other landing path as ignored.
- **Hand-landed or governed:** none (`scripts/owner_land.py:25-28`, `GOVERNED_FILES` in `scripts/gate12_parity_gate.py:174`).
- **Not in this landing:**
  - the registry and manifest entry for `--c13-model-loader`: a draft exists in `docs/drafts/loop_v1/switch_patches_20261123/`, and it waits for registry-repair (owner ruling 17:45);
  - LL-7 and LL-8.

## 3. Results

One suite at a time, under nohup, polled in the foreground (`data/ll6_prep/run_P.sh`, `a4868ea2…`). Settings: `taskset -c 0-2 nice -n 10`, `python3 -u -B`, `PYTHONDONTWRITEBYTECODE=1`, `CUDA_VISIBLE_DEVICES=""`, venv `~/venvs/torch`. Summary `data/ll6_prep/P_summary.txt` (`e1300dfb…`).

| Suite | Result | Before LL-6 (post-LL-landing check, 7f4505a) | log sha256 |
|---|---|---|---|
| `tests/test_ll6_inference_bundle.py` (fresh fixtures) | **12/12 gates, 11/11 mutants**, 143 s (fixtures 18 s), RSS 597 MB, largest child 626 MB | new | `8ad18450…` |
| `tests/test_ll_feature_settings.py` | 51/51, 15/15 | 51/51, 15/15 | `dcde2b4e…` |
| `tests/test_ll_feature_settings.py --runh-root …/e2e-map/data/e2e_runH` (reads run H read-only, as before) | 53/53, 15/15 | 53/53, 15/15 | `b359fb9c…` |
| `tests/test_e2e_vote_harness.py` | 15/15, 17/17; G19 with = this tree; M25 killed | 15/15, 17/17 | `764c4bb8…` |
| `tests/test_e2e_vote_ll5.py` | 3/3, 3/3 | 3/3, 3/3 (pre-landing proof) | `40c5076b…` |
| `tests/test_e2e_vote_ordering.py` | 6/6, 3/3 | 6/6, 3/3 (pre-landing proof) | `a30657e1…` |
| `tests/test_e2e_miss_report.py` | 7/7, 5/5 | 7/7, 5/5 (pre-landing proof) | `bdde1f13…` |
| `tests/test_step1_screen.py` | GATES 16/16, MUTANTS 15/15, SENTINEL PASS | 16/16, 15/15 | `9d315e26…` |
| `scripts/diagnostics/e2e_walk_forward_selftest.py` | **108/108**; sides without = `50cd6837…`, with = this tree | 108/108 | `09847993…` |

**The LL-6 gates** (`data/ll6_P_selftest/selftest_summary.json`):

| Gate | What it shows | Mutants that turn it red |
|---|---|---|
| `LL6.baseline_precondition` | baseline `7f4505a` lacks LL-6, this tree has it. Negative check: a "tree under test" made of the baseline's own files fails it (`data/ll6_prep/neg_precondition.log`); `--baseline-rev HEAD` is refused | (negative check) |
| `LL6.step6_scaler_identical_to_base` | NN: the same flag, mean and scale arrays and the same 3 log lines as `7f4505a`'s `_load_nn_scaler` | `scaler_log_changed`, `scaler_flag_dropped`, `clamp_threshold_changed`, `step6_flag_not_set` |
| `LL6.step6_cli_default_output_identical_to_base` | **four families, real Step 6 `main()`:** 35 of 35 keys equal, all log lines equal, 10 predictions each. Only `agent_metadata/timestamp_injected` and `metadata/timestamp` differ, as they also do between two baseline runs | `scaler_flag_dropped`, `clamp_threshold_changed`, `step6_flag_not_set` (**non-vacuity**: Step 6 output changes) |
| `LL6.A-NORM_all_four_families` | Step 6 against the attribution path's own forward pass: NN 3.67e-9, xgboost, lightgbm and catboost 0 (atol 1e-5) | `bundle_transform_skips_scaler`, `bundle_names_reversed` |
| `LL6.relocated_dir_bundle_keeps_scaler` | measured, Step 6 unchanged: with a moved NN directory Step 6 loses the scaler; the default bundle (`resolved`) keeps it (gap 1.89); `as_written` reproduces Step 6 (0) | `scaler_source_default_as_written` |
| `LL6.c13_legacy_loader_identical_to_base` | four families equal to `7f4505a` (NN `None` = Blocker 3, as before) | (covered with the next one) |
| `LL6.c13_default_construction_loads_as_base` | the loader a caller gets by passing nothing loads what `7f4505a` loads | `c13_default_canonical` |
| `LL6.c13_default_is_legacy` | constructor and CLI default `legacy`; choices `legacy`, `canonical` | `c13_default_canonical` |
| `LL6.c13_canonical_loader_all_four` | bundle, attribution object and the sidecar's names for all four; **non-vacuity**: differs from the baseline's load for all four | `bundle_names_reversed`, `c13_canonical_names_dropped` |
| `LL6.c13_root_cause_legacy_identical_to_base` | `bundle=None`: root-cause result equal to `7f4505a` apart from its timestamp; archives under `--out` | `c13_legacy_input_rerouted` |
| `LL6.c13_root_cause_canonical_uses_scaled_named_input` | NN: all 20 attribution inputs equal `bundle.transform(bundle.vector(features))` | `c13_input_unscaled` |
| `LL6.no_write_outside_out` | `git status --porcelain --ignored --untracked-files=all` outside `data/` unchanged | (by construction) |

- **How each mutant is proved:**
  - its anchor occurs exactly as often as declared, and the file's sha changes;
  - a marker line placed before the mutated statement is touched, where a statement position exists (7 of 11; the other 4 mutate a default argument or an expression);
  - every target gate is red under it.
- **Afterwards** (`data/ll6_prep/P_after.txt`, `00d41273…`):
  - no driver or test process left;
  - 0 new `__pycache__` outside `data/` in this tree, the main tree, e2e-map, ch14-learn, e2e-vote and e2e-grader;
  - Optuna listing unchanged (125 entries);
  - main tree `git status` the same before and after the runs;
  - the `_records` copy verifies.
- **Development runs:** `data/ll6_dev1` to `ll6_dev3` (logs `data/ll6_prep/dev*.log`) found and fixed two faults in the test before the final run:
  - mutated copies were not imported, because `prediction_generator.py:57` puts its own directory first on `sys.path`; the test now imports the bundle and Chapter 13 first and checks which file each import resolved to;
  - an NN A-NORM gap of 1.65e-2 came from forwarding the bundle's module in training mode (§5).

## 4. What it changes for pools, or for how hits against k/1000 are measured

**Nothing at defaults.**
- Step 6 output is unchanged (gate above), so pools are unchanged. Chapter 13's default loader is unchanged.
- `--c13-model-loader canonical` changes only which model object Chapter 13's observe-only root-cause report attributes with, and on what inputs. It trains, ranks and grades nothing.

## 5. Limits and open items

- **The NN is tested on a 200-row CPU fixture only,** with one Step 5 trial per family. A-NORM has not been run on a real run H model on this tree (the W2 note did block 0 on the branch).
- **The bundle hands over the NN module in training mode** (`load_inference_bundle` → `NeuralNetWrapper.load`; `model.training` is `True` after load).
  - Step 6's `predict` (`models/wrappers/neural_net_wrapper.py:350`) and Chapter 13's NN attribution (`per_survivor_attribution.py:91`, restored at `:121`) both call `eval()` first, so neither production path is affected.
  - A new consumer that forgets `eval()` gets dropout and BatchNorm batch statistics. A plain forward in training mode also updates BatchNorm's running statistics. Not changed here; Beta question Q3.
- **Canonical plus name-keyed predictions: the regime-shift tier comparison fails non-fatally.** `_run_regime_shift_analysis` (`chapter_13_orchestrator.py:859`) passes the raw predictions to `compare_pool_tiers` (`:873`), and with dict features that raises "float() argument must be … not 'dict'". It is logged as a warning (`:904`) and the tier file is not written. The per-survivor attribution itself goes through the bundle (gate above). Seen in the fixture run; not changed here; Beta question Q4.
- **Step 6 relocated-directory behaviour is unchanged on purpose:** Step 6 still loses the scaler when the NN directory moves (measured gap 1.89 in the fixture). Only the bundle's default (`resolved`) differs from Step 6.
- **No walk-forward fingerprint comparison was run for this landing.** The walk-forward selftest passes 108/108, but its fingerprint path runs lightgbm on CPU, which never calls the NN scaler. The Step 6 CLI gate covers all four families.
- **Registration of `--c13-model-loader`** waits for registry-repair (owner ruling 17:45). Until then the setting exists only as a CLI flag and a constructor argument.
- **No hit-rate evidence.** Nothing here was graded.

## 6. Draft questions for a Beta review (pipeline code: Beta reviews before commit)

Not sent; the lead decides. Tree: `~/tfm_worktrees/ll6-land` (detached `7f4505a`, the 5 files above).

- **Q1. Default unchanged, Step 6.**
  - Is moving `_load_nn_scaler`'s body into `models.inference_bundle.load_nn_scaler` behaviour-preserving for every input?
  - Read side by side: `self._x =` became a returned dict, `self.logger` became `logger`, numpy is imported at module level.
  - Is `LL6.step6_cli_default_output_identical_to_base` enough evidence (four families, real `main()`, 35 keys, logs, control repeat)?
- **Q2. Blocker 3 and A-NORM.**
  - Does `--c13-model-loader canonical` meet the system map's ruling (`:95-111`, `:256`): the canonical loader in Chapter 13, the exact NN scaler, and A-NORM within a declared tolerance?
  - Is an independent forward through `bundle.attribution_model` the right definition of "attribution-path forward prediction" while LL-7 is not yet landed?
- **Q3. Training mode.** Should `load_inference_bundle` return the NN module in eval mode? Or should every attribution backend keep calling `eval()`, as `per_survivor_attribution.py:91` does today?
- **Q4. Tier comparison.** With canonical loading and name-keyed predictions, `compare_pool_tiers` gets raw dict features and fails non-fatally. Route it through the bundle in this landing, or record it for LL-7/LL-8?
- **Q5. `scaler_source`.** The bundle defaults to `resolved`; Step 6 behaves as `as_written` and loses the scaler for a moved NN directory. Keep Step 6 unchanged and record it, or open a separate Step 6 fix behind its own switch?
- **Q6. Test standard.** Pinned full-sha baseline with a precondition gate, a Step 6 CLI comparison with a control, and source mutants in a `sys.path`-first code dir, each with a marker where possible. Is anything a landing of this class should also cover?
