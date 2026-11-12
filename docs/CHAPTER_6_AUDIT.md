# CHAPTER 6 AUDIT — Anti-Overfit Training (Step 5) vs. source

**Repo:** `/home/michael/distributed_prng_analysis` on VM 101, code base `a15b5247336255801de8cf67d52653701b1e8c80` (`a15b524`).
Current HEAD is `741822b`; `git diff --name-only a15b524 741822b` touches only `docs/`, so the code at `741822b` is
identical to `a15b524`.
**Working-tree caveat (recorded, not mine):** during this pass `git status` showed uncommitted changes to
`agents/watcher_agent.py` (+86/−2) and `agents/watcher_dispatch.py` against HEAD, and a new untracked
`docs/CLAUDE_CODE_INSTRUCTIONS_WATCHER_SAFETY_REPAIRS_20261111.md`. They are another session's. Every WATCHER anchor
below was therefore read from the committed blob (`git show a15b524:agents/watcher_agent.py`), not from the working
tree. Every other cited file is byte-identical to `a15b524` (`git diff --stat a15b524 -- <files>` empty).
**Date:** 2026-11-11. Read-only. Nothing was trained, run or dispatched. "Works" below means the code path is complete
on reading.
**Audited:** `docs/CHAPTER_6_ANTI_OVERFIT_TRAINING.md` (738 lines at `a15b524`).
**Numbering:** bare "Step N" = the **executable** scheme (`docs/PIPELINE_BEHAVIOUR_MODEL.md` §1.1). Step 5 =
`meta_prediction_optimizer_anti_overfit.py` (`agents/watcher_agent.py:387-395` `STEP_SCRIPTS[5]` at `a15b524`;
`agent_manifests/reinforcement.json:4` `"pipeline_step": 5`). The chapter's title uses the same scheme; its §3.2
diagram labels the sieve "STEP 2", which is the conceptual scheme (D21).
**Framing held throughout:** TFM is functional mimicry of PRNG surface output. Nothing here assumes a "true seed".

Abbreviations: `mpo` = `meta_prediction_optimizer_anti_overfit.py`; `tst` = `train_single_trial.py`;
`fsw` = `full_scoring_worker.py`; `ss` = `survivor_scorer.py`; `hq` = `holdout_quality.py`; `pg` = `prediction_generator.py`;
`mf` = `models/model_factory.py`; `wa@` = `agents/watcher_agent.py` at `a15b524`; `rj` = `agent_manifests/reinforcement.json`;
`CH6` = the chapter.

---

## 1. Verdict

```
ACCURATE                 21 claims
BROKEN                    8 claims   (0 already recorded in the governance trail)
SWITCHED-OFF-BY-RULING    2 claims
DESCRIPTION-ONLY         21 claims
UNVERIFIED                2 claims
Absorbed into an item     6 claims   (counted once, inside the item named in §9)
---------------------------------------
Total checked            60 claims
```

Null results first, each with its denominator:

1. **The 20 % test split is never evaluated.** 0 of the 4 model families' final models is scored on it; every
   "overfit" metric the chapter describes is a constant (B1).
2. **The K = 20 folds WATCHER requests reach 0 of the compare-models children.** Host logs: 15 of 15 compare-mode
   children ran 5 folds; 1 of 1 single-model run ran 20 (B2).
3. **Optuna's best architecture reaches 0 of 1 final NN checkpoints on the host** (sidecar records 84-45-35-32;
   checkpoint is 256-128-64) (B3), and CatBoost's two sampled structural parameters reach 0 trials (B4).
4. **Step 6 runs 0 schema-hash comparisons on the production path** (B5).

Short version: the training **target** the chapter prescribes (`holdout_hits`) was replaced under an approved proposal
(S1); the anti-overfitting **apparatus** the chapter is named for exists as definitions whose inputs are constants;
and three of the parameters the search or the operator chooses do not reach the model that is saved.

---

## 2. The chapter-specific questions

### Q1. Which step is this, and in which scheme?

Executable Step 5 (see header). The chapter's §11.1 sequence 3 → 4 → 5 → 6 is accurate (`wa@:387-395`).

### Q2. The training target — re-derived from code

- Step 5 loads `y` from the record key `holdout_quality`: `mpo:1479-1483` (`target_field='holdout_quality'`), loader
  default `mpo:612`, top-level lookup `mpo:657-658`. Excluded from `X`: `['score','confidence','holdout_hits','holdout_quality']`
  (`mpo:1482`); only the first two are keys of the `features` dict, so 91 − 2 = **89** columns train.
- Step 3 writes `holdout_quality` at top level: `fsw:423` (batch path), `fsw:484` (fallback).
- Composite (`hq:99-125`): 0.50 × [0.40 `residue_1000_match_rate` + 0.20 `lane_agreement_8` + 0.20 `lane_agreement_125`
  + 0.20 `lane_consistency`] + 0.30 × [0.34 / 0.33 / 0.33 coherence at mod 1000 / 8 / 125] + 0.20 × `temporal_stability_mean`,
  clamped to [0, 1].
- **How the holdout features are generated** (not in the chapter): `fsw:412-418` calls
  `extract_ml_features(seed, lottery_history=holdout_history, skip=get_survivor_skip(meta))`; `get_survivor_skip` takes
  `skip_best`, then `skip_min`, then 0 (`hq:47`). No tracked `*.py` outside `hq` produces `skip_best`
  (`/bin/grep -rn skip_best --include=*.py`), so the value is `skip_min`, which under the canonical record is the
  trial's lower skip bound (`utils/canonical_records.py:226`). `ss:355` → `_generate_sequence` → the CPU reference, where
  `skip` is a one-time burn before `n` consecutive outputs (`prng_registry.py:176-177`, `java_lcg_cpu`). So the target's
  holdout features compare outputs from position `skip_min` onward with the holdout draws, **not** from position
  `len(train_history)`. The continuation law `offset = len(train_history)` applies to `holdout_hits` only
  (`fsw:297-308`, `:367`). This change is recorded as approved: "Skip semantics | Uniform 4000 | Per-survivor from NPZ"
  (`docs/S111_IMPLEMENTATION_PLAN_FINAL.md:481`; approval `:6`, `:445`). See S1 and open item O1.
- `holdout_hits` is still computed and written (`fsw:364-367`, `:407`) and excluded from `X`.
- Stale `holdout_hits` wording remains: loader docstring `mpo:620`; `rj:17`, `:43`; `agent_manifests/definitions.json:185`.
- Selfplay trains on a different objective: committed Chapter 13 audit **B13** (R² fitness, `holdout_hits` target).
  Cross-referenced, not re-reported.

### Q3. Y-normalisation (S121) — re-derived

NN only. `tst:497-512`: `y_mean`, `y_std` from `y_train` (guard `y_std < 1e-8 → 1.0`), both splits z-scored. Predictions
are inverse-transformed before metrics so R² is on the original scale (`tst:687-701`). Both values are saved in the
checkpoint (`tst:717-719`) and applied at inference by `NeuralNetWrapper.predict` (`models/wrappers/neural_net_wrapper.py:357-359`,
loaded `:437-441`). Tree models train on raw `y` (`mpo:998-1140`). Host checkpoint: `y_mean 0.27418`, `y_std 0.0020068`
(pickle disassembly, §8) — the target spans a very narrow band.

### Q4. Feature count — re-derived

| block | n | anchor |
|---|---|---|
| match / stats / lanes / S119 digits / temporal (incl. `score`, `confidence`) | 27 | `ss:702-731` |
| residue, mods 8 / 125 / 1000 × {match_rate, kl_divergence, coherence} | 9 | `ss:624-667`, merged `:734` |
| Battery Tier 1A | 23 | `ss:754-766` |
| Step-3 metadata merge list | 18 | `ss:774-779` |
| `global_*` | 14 | `fsw:403-405`; `models/global_state_tracker.py:366` (14 names) |
| **extracted** | **91** | |
| minus `score`, `confidence` (`mpo:1482`) | **89 trained** | host sidecar `feature_schema.feature_count = 89`, 89 names, 9 `residue_*`, 14 `global_*`, 23 `batt_*` |

Agrees with committed Chapter 14 audit Q3. The chapter's 47 (§3.2) and 62 (§2.1, §9.2, §11.2, Summary) are D4.

### Q5. `definitions.json` (gitignored, read on the host)

`agent_manifests/definitions.json`, 6,635 B, mtime 2026-01-01, `schema_version 1.1.0`, `updated_at 2026-01-02`, ignored by
`.gitignore:41`. For Step 5 it declares the sidecar blocks `signal_quality`, `data_context`, `feature_schema` (`:14-56`) —
all three are written (`mpo:1293-1295`). It declares that Step 6 validates `feature_schema_hash` and treats a mismatch as
`fatal_error` (`:74-78`, `:91-94`) — the production path does not (B5). Its fingerprint input format includes
`{prng_type}|{mod}` (`:157`), which `mpo:482-491` deliberately excludes since v3.3. Its sidecar example shows
`feature_count: 62` and `target_name: "holdout_hits"` (`:178`, `:185`). **No tracked `*.py` or `*.sh` reads it**
(`git ls-files '*.py' '*.sh' | xargs /bin/grep -l 'definitions\.json'` → none). It changes no label.

### Q6. RANGE-MINER, the 22-array NPZ, the window anchor

None changed Step 5's code (`mpo`, `tst`, `hq` have no reference to the miner, the NPZ contract or the anchor). They
changed what Step 5 trains on: 13 trial-level columns (`utils/canonical_records.py:216-244`, copied per seed `:246-257`),
the per-seed sieve match rates absent from `X` (`ss:774-779`), skip-shape columns zero. Canonical array 4 `offset` is now
the window anchor (`utils/canonical_records.py:217-225`) and is not a model input. Detail in CH6 §13; governed holds in S2.

---

## 3. What actually works end to end today

Each item is a complete code path on reading. None was run.

1. **WATCHER Step 5 command.** `rj:173-184` `default_params` → `wa@:1815` step-scoped merge → `wa@:2141-2154` flag
   construction. Host log (`logs/s120_step4_6.log`): `--compare-models --trials 20 --output-dir models/reinforcement
   --enable-diagnostics --model-type catboost --enable-vmap --k-folds 20`.
2. **Compare-models runner.** `--compare-models` without `S88_COMPARE_MODELS_CHILD` → `_s88_run_compare_models`
   (`mpo:3112-3130`): skip-registry filter (`mpo:150-185`), one child process of `mpo` per model type
   (`mpo:226-269`; LightGBM forced to CPU `:265-266`, `:1052`), archive per model under
   `models/reinforcement/compare_models/<run_id>/<model>/` (`mpo:193`, `:274-278`), winner = highest
   `training_metrics.r2` in the child's sidecar (`mpo:94-139`, `:288-311`), winner's files copied back to
   `models/reinforcement/` (`mpo:313-321`), summary `diagnostics_outputs/compare_models_summary_<run_id>.json` (`mpo:325`).
3. **Single-model child.** Load, signal quality, data context (`mpo:1474-1507`); unseeded 80/20 split (`mpo:1509-1519`);
   degenerate early exit with sidecar, exit code 2 (`mpo:1554-1589`, `:3194-3209`); `n_trials > 1` → Optuna
   (`mpo:2726-2728`), else single-shot (`mpo:2730-2780`).
4. **Optuna.** TPE seed 42, direction **maximize**, per-study SQLite or journal storage (`mpo:1737-1780`); objective = mean
   R² over `KFold(shuffle=True, random_state=42)` folds of the 80 % train+val set (`mpo:2005-2121`); NN folds through
   `tst` subprocesses, tree folds inline (`mpo:2105-2116`); NN parallel across detected GPUs (`mpo:1746-1755`).
5. **Final model.** Retrained on the first 80 % of train+val, validated on the last 20 % (`mpo:1818-1838`); NN via `tst`
   with `--save-model` (`mpo:2540-2720`), tree inline (`mpo:1832-1836`).
6. **NN training.** StandardScaler (zero scale → 1.0) and LeakyReLU forced on (`mpo:244-246`, `:1879-1880`, `:2568-2570`;
   `tst:486-492`), y-normalisation (Q3), early stopping on validation loss, best state restored (`tst:621-679`),
   enriched checkpoint (`tst:708-725`).
7. **Sidecar.** Always written: success (`mpo:1286-1324`, `:2898-2938`) or degenerate (`mpo:2966-3019`).
8. **Step 6 gate on the sidecar.** `signal_quality.prediction_allowed` read before loading (`pg:308-347`); model loaded via
   `load_model_from_sidecar` (`pg:350-353`); input columns ordered by the sidecar's `feature_names` (`pg:366`, `:797-798`).
9. **WATCHER completion test.** `evaluation_type: file_exists` on `models/reinforcement/best_model.meta.json` (`rj:166-168`).

---

## 4. BROKEN

"Governed" means already recorded in the governance trail. Searched for every item: `docs/TB_RULING_*`, `docs/PROPOSAL_*`,
`docs/TEAM_ALPHA_*` (140 files), the fact file, `TODO_CONSOLIDATED.md`, `PROJECT_FILE_CATALOG.md`, `BACKLOG.md`, the
S100/S101/S111/S121 changelogs. **None of the eight is recorded there.**

**B1 — The holdout test set is created and never evaluated; every overfitting metric is a constant.**
- Chapter: `test_size=0.2` "TRUE test set" (§6.1), `final_evaluation()` "Holdout test set evaluation" (§6.2),
  `is_overfitting()` on `overfit_ratio`, `test_mae`, `p_value` (§7.3), sidecar `test_mae`, `overfit_ratio` (§9.2).
- Code: `_create_splits` builds `X_test`/`y_test` (`mpo:1509-1519`); they are never read again (`/bin/grep -n 'X_test\|y_test' mpo`
  → only `:1518-1519`, `:1530`). `_compute_final_metrics` returns `train_variance = val_variance = test_variance = 0.0`,
  `train_mae = sqrt(train_mse)`, `val_mae = sqrt(val_mse)` (RMSE under an MAE name), `test_mae = 0.0`, `overfit_ratio = 1.0`,
  `variance_consistency = temporal_stability = 1.0`, `p_value = 0.01`, `confidence_interval = (0.0, 1.0)` (`mpo:2782-2797`).
  With those inputs `is_overfitting()` (`mpo:1352-1358`) is always False, so `main` always prints "Model generalizes well
  to test set!" (`mpo:3219-3223`). `composite_score()` (`mpo:1360-1371`) has no caller. No method `final_evaluation`
  exists. History: the placeholder values are present since the initial commit (`git log -S'p_value=0.01'` → `0101306`).
- Host: the sidecar carries `test_mae` and `overfit_ratio` keys (§8).
- Stops at: `mpo:2782`.
- Repair would have to: score the saved model on `X_test`/`y_test`, compute the overfit quantities from real errors, and
  write them to the sidecar; or remove the claim and the message.

**B2 — The `k_folds` WATCHER passes never reaches compare-models training.**
- Chapter: `--k-folds` "K-fold cross-validation splits" (§10.2). Manifest: `k_folds: 20`, "Team Beta approved 2026-02-20"
  (`rj:116-122`, `:183`).
- Code: WATCHER emits `--k-folds 20` (item 3.1). With `--compare-models` (`rj:176`) the parent hands off to
  `_s88_run_compare_models` before parsing anything else (`mpo:3112-3130`); the dict it passes has no `k_folds`, and the
  child command is only `--survivors --lottery-data --model-type --trials [--enable-diagnostics] [NN flags]` (`mpo:230-259`).
  The child therefore uses the argparse default 5 (`mpo:3047`). `--output-dir`, `--enable-vmap` and `--persistent-workers`
  are dropped the same way.
- Host evidence: `logs/s112_step5_4models.log` — 12 of 12 children "K-Fold CV: 5 folds" under a parent `--k-folds 20`;
  `logs/s120_step4_6.log` — 3 of 3. Single-model run `logs/s121_ynorm_smoke.log` — 1 of 1 "20 folds". The S100 record
  "Both model types confirmed receiving `--k-folds 20`" (`docs/SESSION_CHANGELOG_20260220_S100.md:62-67`) cites the parent
  `EXEC CMD`; it is not supported by these child logs.
- Stops at: `mpo:230`.
- Repair would have to: forward every Step-5 manifest parameter that the single-model path consumes to each child, and
  show a child log line carrying it.

**B3 — The final NN model is not trained with Optuna's best architecture or patience.**
- Chapter: §8.1 samples architecture; §6.2 "save_best_model()" saves the result.
- Code: `self.best_config = dict(study.best_trial.params)` (`mpo:1813`) — the flat study keys `n_layers`, `layer_0…`,
  `patience`, `dropout`, `learning_rate`, `batch_size`, `epochs`. It is passed as `--params` (`mpo:1826-1829`, `:2576`).
  `tst` reads `hidden_layers` (default `[256, 128, 64]`, `tst:522`) and `early_stopping_patience` (default 15, `tst:601`),
  neither of which the flat dict contains. Trials themselves are correct: `_sample_hyperparameters` returns the list form
  (`mpo:2462-2477`). Since S89 (`git log -S'dict(study.best_trial.params)'` → `9676bca`).
- Host: sidecar `hyperparameters` = `n_layers 4`, `layer_0..3 = 84, 45, 35, 32`, `patience 15`; checkpoint
  `models/reinforcement/best_model.pth` pickle has `hidden_layers [256, 128, 64]` and first weight tensor 91,136 B = 89 × 256 × 4
  (read by `unzip -l` and `pickletools` disassembly; nothing unpickled).
- Stops at: `mpo:1813`.
- Repair would have to: rebuild the training config from the best trial (or re-run `_sample_hyperparameters` on a
  `FixedTrial`) before the final fit, and show the checkpoint's `hidden_layers` equals the sidecar's.

**B4 — CatBoost's sampled `iterations` and `depth` are ignored in every trial and in the final model.**
- Chapter: §8 "Hyperparameter Sampling" (CatBoost space not shown; claim is that sampled parameters drive training).
- Code: the CatBoost space returns `iterations`, `depth`, `learning_rate` (`mpo:2497-2502`); `_train_catboost` reads
  `params.get('n_estimators', 200)` and `params.get('max_depth', 8)` (`mpo:1105-1106`), so every trial and the final model use
  200 × depth 8 with only the learning rate searched. `tst:381-382` has the same key mismatch. The sidecar then records
  the sampled, unused values (`mpo:1813`, `:2843`). CatBoost is the manifest default (`rj:180`). Since S89 (`9676bca`).
- Stops at: `mpo:1105`.
- Repair would have to: one key vocabulary between the search space and the trainer, and a sidecar that records the
  parameters the model was built with.

**B5 — Step 6 does not validate the feature-schema hash.**
- Chapter: "Step 6 validates this hash before making predictions" (§9.3). Also `definitions.json:77`, `:91-94`.
- Code: `pg:350-353` calls `load_model_from_sidecar(models_dir=..., device=...)` without `survivors_file`; the hash and count
  checks run only `if survivors_file and strict` (`mf:334`). Even when run, the count check reads `total_features` or
  `per_seed_feature_count` (`mf:344`), which Step 5 does not write (it writes `feature_count`, `mpo:592`). What Step 6 does
  instead: order input columns by the sidecar's `feature_names` (`pg:366`) and fill any missing name with 0.0 (`pg:798`).
  `validate_feature_schema_hash` is imported (`pg:68`) and not called. A renamed Step-3 feature becomes a silent zero
  column, not a fatal error.
- Stops at: `pg:350`. This is Chapter 7's surface; recorded here because Chapter 6 makes the claim.
- Repair would have to: compare the runtime schema hash (or name list) against the sidecar before prediction and fail
  on mismatch.

**B6 — `--study-name` and `--storage` have no effect.**
- Chapter: `--study-name` "Optuna study name", default `anti_overfit` (§10.2).
- Code: parsed (`mpo:3051-3054`), stored as `self.study_name`/`self.storage` (`mpo:1440-1441`), never read again
  (`/bin/grep -n 'self.study_name\|self.storage' mpo` → `:1440-1441` only). Optuna names the study
  `step5_{model}_{schema_hash[:10]}_{fingerprint[:10]}[_catb22]` and picks its own storage (`mpo:1737-1767`).
- Stops at: `mpo:1440`.
- Repair would have to: honour the flags or remove them.

**B7 — Step 5 does not read Step 4's output.**
- Chapter: `reinforcement_engine_config.json` from Step 4 supplies "Derived training parameters" (§11.2); Chapter 5 §2.4
  says the same.
- Code: no occurrence of `reinforcement_engine_config` in `mpo` or `tst`. The manifest lists it as a required input
  (`rj:9`), which WATCHER uses only to locate inputs for the freshness/preflight check (`wa@:783-800`); no flag carries it to
  `mpo` (`rj:173-184`). Step 6 is the reader (`--config reinforcement_engine_config.json`, host log `logs/s120_step4_6.log`).
- Host: `reinforcement_engine_config.json` does not exist in the working tree now (`ls` → no such file); it is gitignored
  (`.gitignore:43` re-includes `*_config.json`, then the file is ignored) and appears in history up to `6ac3312`.
- Stops at: `mpo` (absence; surfaces listed in §8).
- Repair would have to: give Step 5 a route for Step 4's capacity decisions, or record that Step 4 informs Step 6 only.

**B8 — The holdout offset is not recorded in any artifact; the sidecar's holdout window is a default, not data.**
- Chapter: "The offset is recorded in metadata for debugging" — `holdout_eval {train_len, holdout_len, offset,
  derived_from}` (§3.2.1).
- Code: no tracked `*.py` writes `holdout_eval`, `derived_from` or `train_history_length`
  (`/bin/grep -rn` over tracked `*.py` → none). Step 3 only logs the offset (`fsw:302`). The nearest sidecar field,
  `data_context.holdout_window`, is computed from `--holdout-draws`, default 1000 (`mpo:3074`, `:476-479`), which WATCHER
  never passes (`rj:173-184`). The window feeds the WATCHER loop-prevention fingerprint (`mpo:486-492`).
- Host: sidecar `holdout_window.draw_count 1000`. (Today's `holdout_history.json` has 3,614 draws but post-dates the
  sidecar, 2026-08-21 vs 2026-03-06, so the two cannot be compared directly.)
- Stops at: `mpo:3074`.
- Repair would have to: record train length, holdout length and offset from the files actually used.

---

## 5. SWITCHED-OFF-BY-RULING

**S1 — `y = holdout_hits` (the chapter's fix) is replaced by `holdout_quality`.**
- Chapter: §3.1, §3.2, §3.3, §6.1 comment, §9.2 `training_target`, §11.2, Summary.
- Recorded authority: `docs/S111_IMPLEMENTATION_PLAN_FINAL.md:6` ("v1.1 proposal APPROVED by Team Beta"), `:445`,
  `:481` (skip semantics "Uniform 4000 → Per-survivor from NPZ"); commits `aaca35d`, `1cb90aa`. The v1.1 proposal file
  itself is not in `docs/` (searched `ls docs | grep -i 's111\|holdout'`). Already recorded: behaviour model §7
  ("SUPERSEDED … DIVERGENT — D5"), Chapter 14 audit Q2. Under the 2026-11-11 authority ruling this approval is a
  recorded review position the owner may keep or lift.
- Code: Q2. `holdout_hits` is still computed and stored (`fsw:364-367`, `:407`) but excluded from `X` and not used as `y`.
- **If lifted:** Step 5 would again train on exact holdout hit rate, measured by continuing each seed from position
  `len(train_history)` — the continuation law. S111 records why it was abandoned: discrete Poisson-like values, R² 0.000155
  (`docs/S111_IMPLEMENTATION_PLAN_FINAL.md:477-480`).
- Absorbed here: the §3.2.1 implication that the target is evaluated at `offset = len(train_history)` (true of
  `holdout_hits`, not of `holdout_quality`), and the §3.3 code snippet.

**S2 — The feature groups the chapter calls per-survivor evidence are trial-level or zero, and the per-seed sieve
evidence is held out of `X`.**
- Chapter: §4.1 "features characterize WHY [each survivor survived]"; §4.2–§4.4, §4.6 predictive-value columns;
  §3.5 expected importances led by `intersection_weight`, `lane_agreement_*`, `skip_entropy`.
- Code: 13 NPZ-backed columns (`intersection_*`, `forward_*`, `reverse_*`, `bidirectional_*`, `skip_min/max/range`,
  `survivor_overlap_ratio`) are built once per trial and copied into every record (`utils/canonical_records.py:216-257`);
  `skip_mean`, `skip_std`, `skip_entropy`, `survivor_velocity`, `velocity_acceleration` are defaulted to 0.0
  (`ss:784-791`); `forward_match_rate`/`reverse_match_rate` are not in the merge list (`ss:774-779`). (The lane features are
  per-seed — Step 3's own regeneration, `ss:612-614`.)
- Rulings: Chapter 14 audit **S5** ("do NOT silently add them", `docs/TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md:167-171`,
  `:276`) and **S6** ("must not begin receiving nonzero values under the existing 89-feature schema", `:199-200`; "Expanding
  RANGE-MINER beyond 22 arrays — Not authorized", `:268`). Cross-referenced, not re-reported.
- **If lifted:** the model would see per-seed sieve match quality and per-seed skip shape, so §4's "why it survived"
  reading could hold; it would need a new feature-schema version, Step-3 merge-list changes, regenerated artifacts and
  retraining (Chapter 14 audit S5/S6 "would also need").

---

## 6. DESCRIPTION-ONLY

| # | Chapter says | Code / current fact |
|---|---|---|
| D1 | Version 3.1.0, "Lines: ~900" (header) | `mpo` is 3,234 lines; self-describes v3.3 / "v3.4" (`mpo:3`, `:33`, `:1655`) |
| D2 | §1.1–§1.3: P ≈ 10⁻¹¹⁹¹, "every bidirectional survivor is significant", "true seed" table, "the sieves already proved that" | τ = 1 bound, not the operating regime; corrected in Chapter 2 §1.3 and §4.3. Seed-identity language replaced by the §1 framing |
| D3 | §2.1 code at "`survivor_scorer.py` line 198" | `ss:365-372` (single seed), `:703` (batch). The equivalence `score = 100 × residue_1000_match_rate` holds at `mod = 1000` because predictions are already reduced mod 1000 (`ss:125`, `:380`) |
| D4 | 47 features (§3.2); 62 (§2.1, §9.2, §11.2, Summary "60/62") | 91 extracted / 89 trained (Q4) |
| D5 | §3.4 potentially circular set {score, confidence, exact_matches, residue_1000_match_rate}; "test both approaches" | Only `score`, `confidence` are excluded (`mpo:1482`); `exact_matches` and `residue_1000_match_rate` train. Under the current target, `residue_1000_match_rate` and the lane/coherence/temporal columns are the training-window versions of the target's holdout-window components (`hq:100-115`; Chapter 14 audit §3) |
| D6 | §3.5, §3.6, Summary: expected importance distribution, ">10 features >1 %", "Holdout Hit@K measurable lift" | Expectations, not code. No Step-5 code computes importance shares or Hit@K. The measured S111 picture differs (Chapter 14 §15, "Feature Signal on Real Data"; `SESSION_CHANGELOG_20260225_S111.md:60-67`) |
| D7 | §4 feature tables | Names accurate except `global_regime_change` → `global_regime_change_detected`; tables omit `temporal_stability_min/max`, `pred_*`, `residual_*`, S119 digits, Battery Tier 1A (host sidecar name list) |
| D8 | §5.1 five models incl. `random_forest` | `mpo` CLI choices are four (`mpo:3057-3059`); compare list is four (`mpo:148`). `random_forest` exists in `tst:785` and `rj:64` only |
| D9 | §5.3 `SubprocessTrialCoordinator.train_model` runs each model via `train_single_trial.py --config` | Isolation is by re-invoking `mpo` once per model type (`mpo:223-269`) and by running NN through `tst` (`mpo:1873-1882`, `:2562-2602`). Tree models train inline inside their child (`mpo:2111-2116`, `:1832-1836`). `SubprocessTrialCoordinator` (method `run_trial`, `subprocess_trial_coordinator.py:211`) is used only on a path the CLI cannot reach (`mpo:758`); `tst` has no `--config` (`tst:845-874`) |
| D10 | §6.1 `__init__(X, y, n_splits, test_size, …)`, study `direction='minimize'` | `__init__(survivors_file, lottery_history, k_folds, test_holdout_pct, …, model_type='catboost', …)` (`mpo:1390-1406`); direction `maximize` (`mpo:1776`) |
| D11 | §6.2 methods `optimize`, `_objective`, `_evaluate_fold`, `final_evaluation`, `save_best_model` | `run` (`mpo:1534`), `_optuna_objective` (`:2005`), `save_best_model` (`:2799`); the other three do not exist (B1 for the evaluation) |
| D12 | §7.2 objective = mean fold `val_mae`, minimised | Mean fold R², maximised (`mpo:2117-2121`, `:1776`) |
| D13 | §8.1 NN space (hidden_layers 1–4, hidden_size 32–256, dropout 0.1–0.5, lr 1e-4–1e-2, weight_decay, batch 32–256, epochs 50–200) | `n_layers` 2–4, `layer_0` 64–256, each next ≤ previous ≥ 32, dropout 0.2–0.6, lr 1e-5–1e-2 log, batch {256, 512, 1024}, epochs 50–200, patience 10–30; no weight_decay (`mpo:2462-2477`) |
| D14 | §8.2 XGBoost space (n_estimators 100–1000, max_depth 3–10, min_child_weight, reg_alpha, reg_lambda) | n_estimators 100–500, max_depth 3–12, lr, subsample, colsample_bytree (`mpo:2479-2486`); LightGBM identical (`:2488-2495`); CatBoost `:2497-2502` (B4) |
| D15 | §9.2 sidecar: flat `feature_schema_hash`/`feature_count`/`feature_names`; `training_metrics` with train/val/test MAE; `provenance.training_target`, `train_history_draws`, `cli_args`, `git_commit`; `.xgb` checkpoint | Schema 3.4.0; names/count/hash nested under `feature_schema`; plus `signal_quality`, `data_context`, `optuna`, `hardware`, `training_info`; provenance keys `run_id, started_at, duration_seconds, survivors_file, n_survivors, model_type, compare_models_used, checkpoint_path, outcome` (`mpo:2873-2928`; host sidecar key tree, §8). No target, git commit or CLI args recorded |
| D16 | §10.2 `--model-type` default `neural_net` | `catboost` (`mpo:3057`); WATCHER also passes `catboost` (`rj:180`); manifest bounds say `neural_net` (`rj:66`) |
| D17 | §10.2 `--save-all-models` flag; table of nine flags | Flag absent; compare mode archives every model by default (`mpo:193`, `:274-278`). Table omits `--test-holdout`, `--storage`, `--device`, `--prng-type`, `--mod`, `--holdout-draws`, `--parent-run-id`, `--enable-diagnostics`, `--normalize-features`, `--use-leaky-relu`, `--dropout`, `--persistent-workers`, `--no-persistent-workers`, `--enable-vmap`, `--allow-inline-nn-fallback` (`mpo:3045-3104`) |
| D18 | §11.2 `holdout_history.json` is a Step-5 input | Step 5 does not read it; holdout information arrives inside `survivors_with_scores.json` as `holdout_quality` (Step 3 reads the file, `fsw:546`) |
| D19 | §11.3 `best_model.xgb`, `best_model.lgb` | `.json`, `.txt` (`mpo:1251-1256`); host directory holds `.cbm .json .pth .txt` and `compare_models/` |
| D20 | Footer "END OF CHAPTER 4" | This is Chapter 6 |
| D21 | §3.2 diagram "STEP 2: SIEVES" | Conceptual scheme. Executable: the sieve runs inside Step 1 (`wa@:387-395`; behaviour model §1.1) |

---

## 7. Stale because of RANGE-MINER, the 22-array contract, or the window-anchor change

1. **Every population-scale number** — "4.3 billion seeds", "831672 survivors", "18000/225 draws" in the sidecar example —
   predates the miner. The host sidecar's survivor count is 85 (§8).
2. **§4's "predictive value" columns** for intersection, skip and velocity features: trial-level or zero under the
   canonical record (S2).
3. **§3.5 expected leaders** `intersection_weight`, `skip_entropy`, `forward_count`: two trial-level, one zero.
4. **`offset`.** The chapter's `offset` is Step 3's holdout continuation (`fsw:300-301`). Canonical array 4 `offset` is now
   the window anchor (`utils/canonical_records.py:217-225`); `best_offset` in `X` is a hardcoded 0 (`ss:707`). Three
   meanings, none interchangeable; `TB_RULING_WINDOW_ANCHOR_SEQUENCING.md:19`, `:53` separates the first two.
5. **`get_survivor_skip`** reads `skip_best` (no producer) then `skip_min`, which the canonical record fills from the
   trial context (`utils/canonical_records.py:226`) — so S111's "per-survivor" skip is per-trial on miner data.

---

## 8. Surfaces searched, and surfaces unavailable

**Searched, in the required order**
- *Governance:* `docs/TB_RULING_*`, `docs/PROPOSAL_*`, `docs/TEAM_ALPHA_*` (140 files) for `k_folds|k-folds`, `iterations`,
  `best_trial`, `test_mae|is_overfitting|_compute_final_metrics`, `study-name|study_name`, `reinforcement_engine_config`,
  `schema_hash|schema hash`, `holdout_quality`, `anti_overfit|meta_prediction_optimizer`; read in part:
  `TB_RULING_WINDOW_ANCHOR_SEQUENCING.md` (`:10-70`), `PROPOSAL_DUAL_GPU_PARALLEL_TRAINING_v2_0.md:14`,
  `PROPOSAL_EPISTEMIC_AUTONOMY_UNIFIED_v1_3.md:428-444`. Then `S111_IMPLEMENTATION_PLAN_FINAL.md` (`:6`, `:205-232`,
  `:445`, `:470-490`), `S111_TEAM_BETA_BRIEFING.md` (grep), `SESSION_CHANGELOG_20260225_S111.md` (`:60-100`, `:170-185`),
  `SESSION_CHANGELOG_20260220_S100.md` (`:55-75`), `HOLDOUT_HITS_IMPLEMENTATION_SUMMARY.md` (`:1-30`), `TODO_CONSOLIDATED.md`
  (Step 5 rows), `PROJECT_FILE_CATALOG.md` (§7, Chapter 6 rows), fact file (grep `k_folds`, Step 4 row).
- *Orientation / chapters:* `CLAUDE_CODE_INSTRUCTIONS_CHAPTER_CORRECTIONS_20261111.md` §0, §1, §3, §5, §8, VIR;
  `PIPELINE_BEHAVIOUR_MODEL.md` §1.1, §6, §7; `CHAPTER_2_BIDIRECTIONAL_SIEVE.md` (header, §1.3–1.4, §4.3, §13–14; not
  edited); `CHAPTER_13_AUDIT.md` (layout, B13, A.10); `CHAPTER_14_AUDIT.md` (§1–§8, addendum headings);
  `CHAPTER_14_TRAINING_DIAGNOSTICS.md` (header, §15–§16 opening, S111/S121 notes); `CHAPTER_5_ML_ARCHITECTURE_OPTIMIZER_v2.md:85-110`.
- *Code (by symbol and line):* `mpo` (whole file in ranges 1–1140, 1140–2131, 2460–3234), `tst` (outline, `:371-400`,
  `:450-780`, main dispatch), `hq` (whole), `fsw` (`:60-95`, `:231-312`, `:380-432`, grep), `ss` (`:95-125`, `:350-390`,
  `:547-800`), `prng_registry.py:170-195`, `:4195-4197`, `models/wrappers/neural_net_wrapper.py:1-140` + grep,
  `models/model_factory.py:263-365`, `models/feature_schema.py:42`, `:175-200`, `models/global_state_tracker.py:366`
  (parsed), `utils/canonical_records.py:210-258`, `subprocess_trial_coordinator.py` (outline), `pg` (grep + `:340-366`),
  `window_optimizer.py:1041`, `:1100-1105`, `wa@` (`git show a15b524:`; `:387`, `:474`, `:783`, `:1815`, `:2080-2160`).
- *Manifests:* `rj` (whole), `/bin/grep pipeline_step agent_manifests/*.json`, `agent_manifests/definitions.json` (whole,
  host, gitignored).
- *Git history:* `git log -S` for `k_folds`, `k-folds`, `p_value=0.01`, `n_estimators', 200`, `"iterations": trial.suggest_int`,
  `dict(study.best_trial.params)`, `holdout_hits`; `git log -- agent_manifests/reinforcement.json` with `compare_models` per
  revision; `git show 7c981fb^:mpo` (`:154-188`, the pre-`holdout_hits` target); `git show e0b55a1 --stat`.
- *Host state (read-only):* `models/reinforcement/` listing; `best_model.meta.json` (key tree, `feature_schema` name list
  and counts, `signal_quality.target_name`, `hyperparameters`, `data_context` windows); `best_model.pth` (`unzip -l`, and
  `pickletools` disassembly of `data.pkl` in the scratchpad — no unpickling, no torch); `logs/s112_step5_4models.log`,
  `logs/s120_step4_6.log`, `logs/s121_ynorm_smoke.log`, `watcher_step5_compare_test.log` (`EXEC CMD` and `K-Fold CV` lines);
  `train_history.json`, `holdout_history.json` (lengths, mtimes); `diagnostics_outputs/holdout_feature_autocorr.json`
  (keys, mtime); `synthetic_train_v2.json`, `synthetic_holdout_v2.json` (exist); `git status --ignored --porcelain`.

**Unavailable — and where it matters**
- **`.127` and the rigs** — not accessed. Every claim is about VM 101.
- **`docs/TEAM_BETA_REVIEW_kfolds_S100.docx`** — binary, not opened (catalog §7 gap 7). If it holds a ruling on
  compare-mode fold forwarding, B2's "no record" is about the readable surfaces only.
- **The S111 v1.1 proposal text** — not in `docs/`; approval cited through the implementation plan.
- **Runtime behaviour** — nothing executed; B1–B6 are traced by source, with host artifacts corroborating B2, B3 and B8.
- **Current `reinforcement_engine_config.json`** — absent on the host, so its contents are unread.

**[UNVERIFIED] items** — §9.

---

## 9. Accurate claims, unverified claims, and absorbed claims (for the count)

**Accurate (21)**

| # | Claim | Anchor |
|---|---|---|
| 1 | This is Step 5 | `wa@:387-395`; `rj:4` |
| 2 | Original target was the training score (circular) | `git show 7c981fb^:meta_prediction_optimizer_anti_overfit.py` `:185-188` |
| 3 | `score` ≡ 100 × `residue_1000_match_rate` at mod 1000 | `ss:365-369`, `:380`, `:388`, `:125` |
| 4 | `holdout_hits` = matches / n_holdout from `prng(seed, n_holdout, skip=offset)` | `fsw:303-308` |
| 5 | `offset = len(train_history)`, derived, never configurable | `fsw:300-302`, `:367`; no `holdout_offset` in tracked `*.py` |
| 6 | `train_history.json` / `holdout_history.json` come from Step 1 | `window_optimizer.py:1100-1105` |
| 7 | `--model-type` / `--compare-models` usage | `mpo:3057-3061` |
| 8 | K-fold with shuffle over the train set | `mpo:2034`, `:2099` |
| 9 | `is_overfitting()` thresholds 1.5 / 1.3 / 0.05 | `mpo:1352-1358` (inert, B1) |
| 10 | Sidecar purpose: load, provenance | `mpo:1286-1324`; `pg:350` |
| 11 | Hash = sha256 of sorted comma-joined names, first 16 hex | `mpo:600-603`, `:586-588` |
| 12 | `agent_metadata.pipeline_step = 5` | `mpo:1312`, `:2920` |
| 13 | `--survivors`, `--lottery-data` required | `mpo:3041-3044` |
| 14 | `--compare-models` trains the four families | `mpo:148` (minus skip-registry exclusions `:150-185`) |
| 15 | `--trials` default 50 | `mpo:3045` |
| 16 | `--k-folds` default 5 | `mpo:3047` |
| 17 | `--output-dir` default `models/reinforcement` | `mpo:3064` |
| 18 | Pipeline position 3 → 4 → 5 → 6 | `wa@:387-395` |
| 19 | `best_model.meta.json` always written | `mpo:1568`, `:2811-2813`, `:2952-3019` |
| 20 | Consumers: `prediction_generator.py`, WATCHER | `pg:350`; `rj:166-168` |
| 21 | Version-history lineage (multi-model v2.0, subprocess isolation, sidecar) | `mpo:6-24` |

**UNVERIFIED (2)**
- **U1** — §2.2 importance split 65.01 % / 34.99 %. The figure is also quoted in
  `docs/PROPOSAL_ML_Architecture_Remediation_v2_0.md:227` (2025-12-30, same author, "PROPOSED"); no run log, model file or
  importance dump behind it was found (`/bin/grep -rl '65\.01'` over the tree: that proposal plus unrelated numeric hits in
  `full_scoring_results/*/chunk_*.json`). Not re-measurable without running the pre-fix target. Would verify: the original
  run's model file or log.
- **U2** — §3.7 synthetic test: seed 12345 → `holdout_hits = 1.0` on `synthetic_holdout_v2.json`. The two files exist on
  the host (untracked, mtime 2025-12-31); the test was not run in this read-only pass. Would verify: running the
  chapter's one-liner (CPU only, no pipeline).

**Absorbed (6), not counted separately**

| Claim | Absorbed into |
|---|---|
| §1.2–§1.3 "true seed / multiple seeds / partial match" table and "sieves already proved" | D2 |
| §3.2.1 implication that the target is evaluated at `offset = len(train_history)` | S1 |
| §3.3 `y = [s['holdout_hits'] …]` | S1 |
| §11.2 "62 features per survivor + holdout_hits" | D4 (count) and S1 (target) |
| §5.3 "This resolves OpenCL/CUDA conflicts…" | D9 |
| Summary "ignores 60/62 features" | D4 |

**Count reconciliation**

| Category | Count | Items |
|---|---|---|
| Accurate | 21 | §9 table 1–21 |
| BROKEN | 8 | B1–B8 |
| SWITCHED-OFF-BY-RULING | 2 | S1, S2 |
| DESCRIPTION-ONLY | 21 | D1–D21 |
| UNVERIFIED | 2 | U1, U2 |
| Absorbed into an item | 6 | table above |
| **Total claims checked** | **60** | 21 + 8 + 2 + 21 + 2 + 6 = 60 |

**Recorded elsewhere, cross-referenced only (not counted):** Chapter 14 audit B1 (`best_round` vs `best_val_round`),
B3 (tree final models get no diagnostics — `mpo:1832`, `:2761`), B5 (`f{i}` / `feature_{i}` names), B18 (NN per-feature
gradients never captured), S1 (step-scoped param filter, so a RETRY does not change Step-5 params — `wa@:1815`), S10
(`actual_mean`/`actual_std` exclusion); Chapter 13 audit B13 (selfplay objective vs `holdout_quality`).

**Not a chapter claim, recorded for the owner (N1):** `SESSION_CHANGELOG_20260225_S111.md:80` records "TB response:
Approved all 6" for guardrails G1–G6 and "Decision: Hold off on S112 patch until clean baseline established" (`:82`).
Chapter 14 audit S10 says no answering ruling was found. G4 (target in study name), G5 (importance in sidecar) and G6
(rank metrics) are absent from `mpo` (`/bin/grep -n -i 'spearman\|feature_importance\|near_constant'` → none;
study name `mpo:1742`). This changes Chapter 14's S10 wording, not a Chapter 6 label.

**Files opened (execution proof):** listed in §8 "Searched". This audit edited nothing but itself and
`docs/CHAPTER_6_ANTI_OVERFIT_TRAINING.md`.
