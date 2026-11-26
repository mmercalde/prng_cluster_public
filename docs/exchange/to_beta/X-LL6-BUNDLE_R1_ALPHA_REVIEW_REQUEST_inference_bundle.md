# X-LL6-BUNDLE  R1  REVIEW_REQUEST  LL-6, one inference bundle for Step 6 and Chapter 13 (before commit)

- **Review class:** BEFORE-COMMIT. This is pipeline code (`prediction_generator.py`, Step 6), so you review before the owner commits. The lead asked for this review in `logs/e2e_session/TASK_WF_POSTLAND_AND_LL6_BETA_20261125.txt` (2026-11-25 19:10). Your review is not binding; the owner decides.
- **Base commit:** `7f4505aaccebdd284393eb3565e75d09d1c0f3b4`. **Main is now `579dbb7`.**
  - `git diff 7f4505a 579dbb7 --stat` changes 7 files: `CLAUDE.md`, `agents/registry/parameter_registry.py`, `docs/LEADS.md`, `docs/drafts/loop_v1/READY_LEADS_TIDY_20261125.md`, `docs/exchange/LEDGER.md` and the two X-LEADS-TIDY R1 exchange files.
  - None is a file under review: all five have the same blob, or are absent, at both commits.
  - None of the five files, nor the modules they import (`chapter_13_*.py`, `per_survivor_attribution.py`, `models/model_factory.py`, `models/wrappers/*.py`), refers to `agents/registry`.
  - `7f4505a` is an ancestor of `579dbb7`. So the base for these paths is unchanged.
- **Rounds:** this is R1. The lead decides whether a second round follows.
- **Allowed answers per question:** `NO OBJECTION` · `OBJECTION` (with `LIVE` or `THEORETICAL`) · `SUGGESTION` · `NEEDS EVIDENCE`.
- **Scope:** real-world cases on this system as it is today: VM 101, Step 6 as WATCHER and the walk-forward run it, and Chapter 13 root-cause analysis with the default and the new setting.

**The files under review are NOT on main.** Read them in `~/tfm_worktrees/ll6-land`, a detached worktree at `7f4505a` with the five files as working-tree changes. The sha256 values are each file's content now.

## Prior record

**Searches run** (main, 2026-11-25):

```
S = docs/PROJECT_FILE_CATALOG.md docs/TB_RULING_* docs/TB_RULING_REQUEST_* docs/PROPOSAL_* docs/CHAPTER_*
    docs/*_AUDIT.md docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/LEADS.md docs/exchange/
    docs/TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md docs/drafts/
git grep -lE P HEAD -- $S   for P in: 'inference_bundle|inference bundle' | 'c13[-_]model[-_]loader' | A-NORM | 'Blocker 3'
/bin/grep -lE 'LL-6|inference bundle|c13-model-loader|A-NORM' logs/e2e_session/*.txt   (logs/ is git-ignored; read on disk)
```

**Documents found that decide, authorise, schedule or record the subject:**
- **The ruling LL-6 implements:**
  - `docs/TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md:95-111`: Blocker 3, "Chapter 13 bypasses the canonical NN loader"; gate A-NORM "(release-blocking): for the same survivor, Step-6 inference prediction == attribution-path forward prediction within a declared tolerance"; and your earlier ruling, quoted there, "One-line Ch13 feature-name patch alone — Not sufficient; do not land in isolation";
  - `:256`: the activation list ("use the canonical sidecar/model loader in Ch13; restore the exact NN scaler; validate A-NORM").
- `docs/CHAPTER_13_AUDIT.md:153-155` (Blockers 2-4) and `:212-214` (what must change).
- `docs/TODO_CONSOLIDATED.md:48`, `:120`: P0-C ("canonical model loader in Ch13, NN scaler, A-NORM parity gate").
- **The lead's order:** `logs/e2e_session/TASK_LOOP_ADDENDUM_3.txt:14` (wave 2: "LL-6 (Chapter 13 loads the model the way Step 6 does, A-NORM gate, all four families, CPU)"); `logs/e2e_session/TASK_LL6_LAND_PREP_20261125.txt` (prepare the landing, LL-6 only).
- **Registration order:** `logs/e2e_session/OWNER_RULING_switch_registration_order_20261125.txt`: registry repair (landed `579dbb7`), then switch registrations, then manifests. The registration of `--c13-model-loader` is not in this landing. Draft entries exist in `docs/drafts/loop_v1/switch_patches_20261123/parameter_registry.py.patch`; `docs/drafts/loop_v1/SWITCH_CHECK_20261123.md` lists the flag.
- **The build record,** on the branch, not on main: `~/tfm_worktrees/ch14-learn/docs/drafts/loop_v1/READY_LOOP_LEARN_W2_20261123.md` (LL-6/7/8, 28 gates on `41ad509`).
- **The landing survey:** `~/tfm_worktrees/e2e-map/docs/drafts/loop_v1/SURVEY_CH14_LEARN_20261125.md` §2 U1, §4.
- **No thread** in `docs/exchange/` reviews LL-6. `X-LL-SWITCHES` reviewed LL-1/2/3, which this landing builds on.

**Labels:**

| Item | Label |
|---|---|
| The LL-6 design: bundle, `--c13-model-loader`, A-NORM | ALREADY RECORDED (system map `:95-111`, `:256`; Ch13 audit `:153-155`) |
| This landing's test file and runner, the full-sha pin, the source mutants | NEW |
| A relocated NN directory loses its scaler in Step 6 | ALREADY RECORDED (W2 note, gate `LL6.relocated_dir_step6_drops_scaler`) |
| The bundle returns the NN module in training mode | NEW |
| Canonical plus name-keyed predictions: the tier comparison fails non-fatally | NEW |
| CHANGES A RULING | none |

**Not searched:** the ser8 archives, the private repository, the rigs, and project files outside the repository.

## Files under review (5)

| # | Lands at | N/M on `7f4505a` | Read it at | sha256 now |
|---|---|---|---|---|
| 1 | `models/inference_bundle.py` | N, force-add (`.gitignore:162` `models/`) | `~/tfm_worktrees/ll6-land/models/inference_bundle.py` | `586a33752363ef5821bbc31b991de1355c5baf3c0b78c8a263f232a9fb904be6` |
| 2 | `prediction_generator.py` | **M** (Step 6) | `~/tfm_worktrees/ll6-land/prediction_generator.py` | `4f6a26bff13d854d928afd623e1a912fb94d93fa8e98cd0dc0793e3e522bd808` |
| 3 | `chapter_13_orchestrator.py` | **M** (Chapter 13) | `~/tfm_worktrees/ll6-land/chapter_13_orchestrator.py` | `40e1fcb326801d6bb3035d981f5ad48f3bf07815bfa8a1b6dba52a6ef5f0906b` |
| 4 | `tests/test_ll6_inference_bundle.py` | N | `~/tfm_worktrees/ll6-land/tests/test_ll6_inference_bundle.py` | `ed703f2f0c2fefe3b58732a72b33eb8509d313a5d11775b16064f6b19610c8ec` |
| 5 | `tests/ll6_step6_runner.py` | N | `~/tfm_worktrees/ll6-land/tests/ll6_step6_runner.py` | `77a910d2f13c1b865489b7bb8611fc6f1b7e44792399cc188a2e85d56b0c5a06` |

- `git -C ~/tfm_worktrees/ll6-land diff --stat` shows `prediction_generator.py` +9 −57 and `chapter_13_orchestrator.py` +57 −4.
- **Where it sits** (ll6-land lines):
  - **Bundle:** `models/inference_bundle.py:55-119` `load_nn_scaler` (Step 6's former body), `:122` `sidecar_feature_names`, `:128` `resolve_checkpoint_path`, `:149-225` `InferenceBundle`, `:228-264` `load_inference_bundle`.
  - **Step 6:** `prediction_generator.py:389-406`. `_load_nn_scaler` now calls `load_nn_scaler` and sets the same three attributes. The rest of the file equals `7f4505a`.
  - **Chapter 13:** `chapter_13_orchestrator.py:85-99` (`C13_MODEL_LOADERS`, `_c13_model_input`); `:131-138` (constructor, default `legacy`); `:586-589` (`legacy` runs the old body); `:674-689` `_load_model_canonical`; `:749-750`, `:773-774` (root-cause inputs: unchanged when `bundle is None`); CLI `--c13-model-loader`, `--c13-models-dir`.
- **How `prediction_generator.py` was made:** main's file plus the LL-6 hunk only, by a clean `git merge-file` (base `41ad509`, ours `7f4505a`, theirs ch14-learn `7368f34`). It is byte-equal to the branch copy, because main's LL-1 hunks equal the branch's.
- **`models/inference_bundle.py` existed in no commit** before this. A copy with a `SHA256SUMS` is in `~/tfm_worktrees/_records/inference_bundle_20261125/`.

**Also to read:**
- **The READY note:** `~/tfm_worktrees/ll6-land/docs/drafts/loop_v1/READY_LAND_LL6_20261125.md`, sha256 `9e95f0787928e95b4f448d38c5fdaafdfbc8e9752e6059021039fea6732649b0`. It lands with the files.
- **Evidence:** `~/tfm_worktrees/ll6-land/data/` (ignored):
  - `ll6_P_selftest/selftest_summary.json` (the final LL-6 run);
  - `ll6_prep/P_*.log`, `P_summary.txt`, `P_after.txt`, `run_P.sh`;
  - `ll6_prep/pg_main_vs_landing.diff`;
  - `ll6_prep/neg_precondition.log`;
  - `ll6_prep/dev*.log` (the three development runs).
- **Post-landing check of main `579dbb7`,** for context only: `~/tfm_worktrees/ll6-land/docs/drafts/loop_v1/POSTLAND_CHECK_579dbb7_20261125.md`.

**About running anything:**
- A write into this repository during your review is a `BOUNDARY VIOLATION`, and so is a write under `~/tfm_worktrees/ll6-land` if you treat it as part of the repository.
- The LL-6 test refuses an `--out` outside `<tree>/data/`. A throwaway copy of the tree under `/tmp` (`AGENTS.md` §3) can run it there; it needs `~/venvs/torch`, CPU only, about 2.5 minutes.
- Please say what you ran and what you only read.

## What was run (VM 101, CPU only, `CUDA_VISIBLE_DEVICES=""`, one suite at a time, `taskset -c 0-2 nice -n 10`, `python3 -u -B`; no commit anywhere)

| Suite (in `ll6-land`) | Result | Before LL-6 |
|---|---|---|
| `tests/test_ll6_inference_bundle.py` (fresh fixtures) | **12/12 gates, 11/11 mutants**, 143 s | new |
| `tests/test_ll_feature_settings.py`, default / `--runh-root` | 51/51, 53/53; mutants 15/15 | the same |
| `tests/test_e2e_vote_harness.py` | 15/15, 17/17 | the same |
| `tests/test_e2e_vote_ll5.py`, `_ordering`, `test_e2e_miss_report.py` | 3/3 3/3, 6/6 3/3, 7/7 5/5 | the same |
| `tests/test_step1_screen.py` | 16/16, 15/15 | the same |
| `scripts/diagnostics/e2e_walk_forward_selftest.py` | 108/108 (with = this tree, without = `50cd683`) | the same |

- **What the LL-6 gates show** (each compares with `git show 7f4505a…:<file>`; `--baseline-rev` must be a full 40-hex sha, and `HEAD` is refused):
  - **precondition:** the baseline lacks LL-6 and the tree under test has it. A negative check with the baseline's own files as the tree under test fails it.
  - **Step 6 scaler:** NN flag, mean and scale arrays, and the 3 log lines are equal to `7f4505a`'s `_load_nn_scaler`.
  - **Step 6 default output, real `main()`, four families** (through `tests/ll6_step6_runner.py`):
    - 35 of 35 keys equal, all log lines equal, 10 predictions each;
    - only `agent_metadata/timestamp_injected` and `metadata/timestamp` differ, as they also do between two baseline runs.
  - **A-NORM, four families:** Step 6 against the attribution object's own forward pass (eval mode). NN 3.67e-9, xgboost, lightgbm and catboost 0; atol 1e-5, declared before measurement in the W2 note.
  - **relocated NN directory:** Step 6 loses the scaler (unchanged, measured gap 1.89); the default bundle (`resolved`) keeps it.
  - **Chapter 13 `legacy`:**
    - loader result equal to `7f4505a` for four families;
    - the default a caller gets by passing nothing loads the same;
    - constructor and CLI default `legacy`;
    - root-cause result with `bundle=None` equal apart from its timestamp.
  - **Chapter 13 `canonical`:**
    - four families load the bundle with the sidecar's names, which differs from the baseline's load (non-vacuity);
    - NN root cause is fed exactly `bundle.transform(bundle.vector(features))`.
  - **no write outside `--out`:** `git status --porcelain --ignored --untracked-files=all` outside `data/` is unchanged.
- **The mutants** are source mutations: one exact replacement in a copy of the file, in a code dir placed first on `sys.path`. Each is shown applied (anchor count, sha change), shown run where a statement position exists (a marker file, 7 of 11), and red on every target gate. Non-vacuity of the Step 6 output gate: `scaler_flag_dropped`, `clamp_threshold_changed` and `step6_flag_not_set` each change Step 6's output.
- **Afterwards:** no new `__pycache__` in any tree; Optuna listing unchanged (125); main tree status the same before and after.

## Known limits (stated so you do not have to find them)

1. **NN only on a 200-row CPU fixture,** one Step 5 trial per family. A-NORM has not been run on a real run H model on this tree. The W2 note reports block 0 on the branch.
2. **The bundle returns the NN module in training mode.** `load_inference_bundle` → `NeuralNetWrapper.load` leaves `model.training == True`, with dropout and BatchNorm active.
   - Step 6 (`models/wrappers/neural_net_wrapper.py:350`) and Chapter 13's NN attribution (`per_survivor_attribution.py:91`, restored at `:121`) both call `eval()` first, so neither production path is affected.
   - A consumer that forgets `eval()` gets dropout and batch statistics, and a training-mode forward updates BatchNorm's running statistics. The first development run of the test did exactly that (NN gap 1.65e-2); the test now uses eval mode.
   - Not changed in the code.
3. **Canonical plus name-keyed predictions: the regime-shift tier comparison is skipped.** `_run_regime_shift_analysis` (`chapter_13_orchestrator.py:859`) passes the raw predictions to `compare_pool_tiers` (`:873`). With dict features that raises "float() argument must be … not 'dict'"; it is logged as a non-fatal warning (`:904`), and `tier_comparison.json` is not written. Seen in the fixture run. The per-survivor attribution goes through the bundle and works. Not changed.
4. **Step 6's relocated-directory behaviour is unchanged** (it reads the scaler from the sidecar path as written). Only the bundle's default (`resolved`) differs.
5. **No walk-forward fingerprint comparison for this landing.** The walk-forward path runs lightgbm on CPU, which never calls the NN scaler. The Step 6 CLI gate covers all four families on fixtures.
6. **`--c13-model-loader` is not registered** in `agents/registry/parameter_registry.py` or any manifest. That follows the owner's 17:45 order and is a separate landing.
7. **No hit-rate evidence.** At default nothing about pools or grading changes. `canonical` changes only Chapter 13's observe-only root-cause report.

## Questions

**Q1. Default unchanged.** With `--c13-model-loader` at `legacy` and Step 6 as WATCHER and the walk-forward run it, do Step 6's output and Chapter 13's root cause behave exactly as on `7f4505a`? That includes the scaler move into `load_nn_scaler`: a returned dict instead of `self._…` attributes, `logger` instead of `self.logger`, numpy imported at module level. Is the evidence enough: four families through the real `main()` with a control repeat, and the in-process scaler comparison?

**Q2. Blocker 3 and A-NORM.** Does `canonical` meet the ruling (`docs/TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md:95-111`, `:256`): the canonical loader in Chapter 13, the exact NN scaler, and A-NORM within a declared tolerance? While LL-7 is not landed, is a forward pass through `bundle.attribution_model` in eval mode the right reading of "attribution-path forward prediction"?

**Q3. The two canonical-path gaps (limits 2 and 3).** Should either be fixed inside this landing: return the NN module in eval mode from the bundle, or route `compare_pool_tiers` through the bundle? Or is recording them, for LL-7/LL-8, enough while `canonical` is opt-in and observe-only?

**Q4. `scaler_source`.** The bundle defaults to `resolved`; Step 6 behaves as `as_written` and loses the scaler when an NN directory is moved. Is leaving Step 6 unchanged, with the difference recorded and measured, correct for this landing?

**Q5. The tests after landing.** Do `tests/test_ll6_inference_bundle.py` and `tests/ll6_step6_runner.py` still test something real once main has LL-6? That covers the full-sha pin with its precondition gate, the `sys.path`-first code dir for mutants, the Step 6 CLI comparison with a control, and the no-write gate. Is any mutant credited without proving its gate, or any gate left without a mutant that should have one?

Please label every objection `LIVE` or `THEORETICAL`, as `AGENTS.md` §5 requires.
