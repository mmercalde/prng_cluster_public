# CHAPTER_CORRECTIONS_REPORT_20261111.md

**Brief:** `docs/CLAUDE_CODE_INSTRUCTIONS_CHAPTER_CORRECTIONS_20261111.md` (REV1), authorized by Michael, 2026-11-11.
**Code base audited:** `a15b524` for all twelve chapters.
- `git diff --name-only a15b524 HEAD` outside `docs/` touches only these files:
  - `agents/watcher_agent.py` and `agents/watcher_dispatch.py`, changed by `b6cb789` and `cf9edaa`;
  - tests: `tests/test_watcher_*`, `tests/test_gate_prod_shape_scoping.py`, `tests/gate_s172_prod_shape.py`.
- Every WATCHER anchor in the batch-2 and batch-3 chapters was taken from `git show a15b524:`.

**Documentation commits for this pass** (made by Michael):
- `741822b`: batch 1, Ch 9, 12, 13, 14.
- `955d0fd`: batch 2, Ch 3, 4, 5, 6, plus the batch-1 count reconciliation.
- `040697f`: batch 3, Ch 7, 8, 10, 11, plus the Ch14 S10 addendum.

**This file and the session changelog are not committed** (stop gate §8).
**Subagent model:** every chapter subagent was launched with `model: opus` set explicitly. From inside, a subagent cannot confirm which model it runs on, which is why several reports say "unconfirmed".

---

## 0. Null results and headline (read first)

- **The pass has 30 SWITCHED-OFF items.** These are designed paths held back by a ruling, an approved spec or an owner decision; see §3. Every one is a decision that belongs to Michael.
- **The pass has 76 BROKEN items, after removing duplicates across all 12 chapters.**
  - 3 were repaired after the audit base, at `b6cb789` (Ch12 B1, B3, B6).
  - **73 are open at HEAD.**
  - 6 of the 73 already have a governance record (marked GOVERNED in §2). They are listed for completeness, not as new findings.
- **Five pipeline-level null results**, each with its denominator:
  - **Chapter 13 has never run on real data.** 233 of 233 archived diagnostics graded an empty pool, all from synthetic draws. Nothing writes `prediction_pool.json`.
  - **Step 6 has made 0 predictions since Step 1's survivor files became count-only stubs** (`8ebdf99`, 2026-04-21). 19 of 19 history files are dated 2026-02-08 to 2026-03-07.
  - **None of the 11 parameters Step 2 tunes reaches any later step.** Downstream code reads only `prng_type` and `mod` (Ch3 B2, Ch4 B5).
  - **Step 4 gave the same answer in 27 of 27 archived runs, and Step 5 reads none of it** (Ch5 B1, B5).
  - **Step 5 never scores its 20% test set.** `test_mae`, `overfit_ratio` and `p_value` are hardcoded (Ch6 B1). All 107 archived Ch14 health checks grade `critical`, because the check reads the wrong key (Ch14 B1).
- **Lead: `323b6fa` ("Doc sync 20260209_1704") silently reverted five chapters.** §7 lists the content that is still missing. Nothing was restored.
- **Pass sentinel: `INCOMPLETE`.** Eleven chapters are `PASS`. Chapter 9 declared `INCOMPLETE` because 8 of its claims need rig access, which the brief forbade (§8).

---

## 1. Per-chapter counts

"Absorbed" means the item is placed under an existing item, in this audit or an earlier committed one, and is not counted separately. In every row the six counts add up to the number of claims checked.

| Ch | Subject (numbering) | Claims | Accurate | BROKEN | SWITCHED-OFF | DESC-ONLY | UNVERIFIED | Absorbed | Chapter lines | Audit file |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 | Scorer meta-optimizer (exec. Step 2) | 74 | 20 | 5 | 1 | 43 | 4 | 1 | 958 → 1594 | `CHAPTER_3_ALIGNMENT_AUDIT.md` 925 → 1211 (addendum) |
| 4 | Full scoring (Step 3) | 83 | 42 | 7 | 2 | 15 | 2 | 15 | 1037 → 1568 | `CHAPTER_4_AUDIT.md` new, 283 |
| 5 | Adaptive meta-optimizer (Step 4) | 41 | 19 | 5 | 1 | 10 | 1 | 5 | 423 → 1022 | `CHAPTER_5_AUDIT.md` new, 293 |
| 6 | Anti-overfit training (Step 5) | 60 | 21 | 8 | 2 | 21 | 2 | 6 | 738 → 1437 | `CHAPTER_6_AUDIT.md` new, 459 |
| 7 | Prediction generator (Step 6) | 47 | 18 | 3 | 1 | 16 | 1 | 8 | 933 → 1668 | `CHAPTER_7_AUDIT.md` new, 413 |
| 8 | PRNG registry (component) | 66 | 35 | 0 | 1 | 11 | 2 | 17 | 1058 → 1525 | `CHAPTER_8_AUDIT.md` new, 295 |
| 9 | GPU cluster + RANGE-MINER | 91 | 41 | 2 | 5 | 35 | 8 | 0 | 980 → 1803 | `CHAPTER_9_AUDIT.md` new, 243 |
| 10 | Agent framework, `_v3` canonical | 66 | 28 | 1 | 1 | 24 | 3 | 9 | 586 → 1065 | `CHAPTER_10_AUDIT.md` new, 369 |
| 11 | Feature importance (component) | 33 | 10 | 4 | 0 | 12 | 1 | 6 | 1099 → 1688 | `CHAPTER_11_AUDIT.md` new, 366 |
| 12 | WATCHER + RANGE-MINER routing | 65 | 24 | 6 | 3 | 27 | 5 | 0 | 880 → 1577 | `CHAPTER_12_AUDIT.md` new, 229 |
| 13 | Live feedback loop | 69 | 28 | 19 | 9 | 9 | 2 | 2 | 1229 → 1843 | `CHAPTER_13_AUDIT.md` 347 → 614 (addenda) |
| 14 | Training diagnostics | 72 | 22 | 18 | 10 | 18 | 2 | 2 | 3199 → 3966 | `CHAPTER_14_AUDIT.md` 369 → 552 (addenda) |
| **Σ** | | **767** | **308** | **78** | **36** | **241** | **33** | **71** | | |

Notes:
- **Ch13 total.** 68 claims were audited, plus B19, which is new. The batch-1 report summed to 63 because six resolved UNVERIFIED items had no final bucket; this is reconciled in `CHAPTER_13_AUDIT.md` §A.10.
- **Ch14 total.** 72 claims. The old U1 *became* B18, and U2 and U3 were absorbed; see `CHAPTER_14_AUDIT.md`, "Count reconciliation".
- **Ch3 changes.** Claim #27 moved from Accurate to Absorbed under Ch4 B5 after a post-review correction (`CHAPTER_3_ALIGNMENT_AUDIT.md` §A.11). The original audit's own summary tally, 17/9/24/5, does not match its own table, which gives 16/5/1/1/27/5 (§A.0 #4).
- **Shrink rule.** Every chapter got longer. No chapter ended more than 10% shorter, so no removal list was needed. All pre-existing headings are present.
- **Line counts.** Ch9 is 1803 here, rather than the 1801 reported at batch 1, because of the B-1 anchor correction.
- **Σ BROKEN.** The 78 counts each chapter's BROKEN items. Removing the 2 cross-chapter duplicates in §2 gives **76**.

---

## 2. Repair list: BROKEN code items, deduplicated

Each line gives the anchor (at `a15b524`) and what a repair must do. No code is given. Status is one of:
- **OPEN**
- **REPAIRED@b6cb789**
- **GOVERNED**: open, and a governance record already exists.

Duplicates merged across chapters (2):
- **Ch14 B13 = Ch13 B15**, the Chapter 13 attribution blockers (P0-C).
- **Ch6 B7 = Ch5 B5**: Step 5 never reads Step 4's output.

Same defect class at different sites, counted per site because each site needs its own repair:
- Ch9 B-1, Ch4 B1 and Chapter 2 F-5: the `rig-6600*` hostname guards.

### 2.1 Cluster and RANGE-MINER (Ch9)

**R-01 (Ch9 B-1).** `distributed_worker.py:18`, `:354`.
- What is wrong: the ROCm prelude only fires on hostnames `rig-6600*`, but the live hosts are `rrig6600*`.
- Repair: follow Ch2 §9.5. Decide whether the prelude is still supported; if it is, key it from the rig profile. **OPEN**

**R-02 (Ch9 B-2).** `persistent_worker_coordinator.py:296`.
- What is wrong: the PWC local slot pool is hardcoded to `range(2)`, but `distributed_config.json:6` says `gpu_count: 1`.
- Repair: take the count from the resolved execution set or the config. **OPEN**

### 2.2 Step 2, scorer meta-optimizer (Ch3)

**R-03 (Ch3 B1).** `run_scorer_meta_optimizer.sh:285`.
- What is wrong: it calls `collect_scorer_results(1)`.
- Repair: pass the real trial count so that a shortfall is reported (`coordinator.py:2591-2593`). **OPEN**

**R-04 (Ch3 B2).** No Step-5 file reads `optimal_scorer_config.json` (`generate_anti_overfit_jobs.py:29-39`).
- Repair: give the four NN parameters a consumer, or have the owner record them as intentionally unconsumed. **OPEN**

**R-05 (Ch3 B3).** `scorer_trial_worker.py:365-366` vs `:368-375`, read at `:421`.
- What is wrong: `UnboundLocalError` on small pools.
- Repair: assign `sm_arr` on both branches. **OPEN**

**R-06 (Ch3 B4).** `run_scorer_meta_optimizer.sh:236-237`.
- What is wrong: a missing line continuation makes the command exit 127. It is reachable through `scorer_meta.json:140-142`.
- Repair: make the legacy branch's argv one command. **OPEN**

**R-07 (Ch3 B5).** `agent_manifests/scorer_meta.json:126`.
- What is wrong: it sets `sample_size: 450` under `arg_style: positional`, but the shell default of 5000 is what runs (`:171`).
- Repair: make the configured and executed values one value. **OPEN**

### 2.3 Step 3, full scoring (Ch4)

**R-08 (Ch4 B1).** `survivor_scorer.py:22`, `:61`; `full_scoring_worker.py:90`, `:99`.
- What is wrong: the ROCm prelude and the 0.8 VRAM cap are keyed to `rig-6600*`, the same class as R-01.
- Repair: key them from the rig profile. **OPEN**

**R-09 (Ch4 B2).** `survivor_scorer.py:373`, `:707`.
- What is wrong: `best_offset` has always been 0, and no alignment search exists.
- Repair: implement the search, or classify the column as dead under P0-A. **OPEN**

**R-10 (Ch4 B3).** `survivor_scorer.py:505`.
- What is wrong: the `batch_score_vectorized` fallback raises `UnboundLocalError`, and its outputs are not reduced by mod.
- Repair: initialise the array and reduce by `self.mod`. **OPEN** (no pipeline caller)

**R-11 (Ch4 B4).** `survivor_scorer.py:802-809`.
- What is wrong: `_cpu_batch_generate` returns raw 32-bit values that are not reduced by mod. 42 of 44 families, and any GPU exception, reach it.
- Repair: reduce by mod, and add the path to F-PARITY. **OPEN**

**R-12 (Ch4 B5).** `full_scoring_worker.py:344`; `generate_step3_scoring_jobs.py:201-209`.
- What is wrong: Step 2's tuned parameters never reach Step 3. On the host the config says residues 17/150/1449, but the output carries 8/125/1000.
- Repair: first decide through governance whether Step 2 should drive Step 3. If yes, pass the parameters, version the schema, and key `holdout_quality` (`holdout_quality.py:61-82`) from the configured mods. **OPEN**

**R-13 (Ch4 B6).** `full_scoring_worker.py:451-456`.
- What is wrong: the sequential fallback merges 6 of 18 metadata fields.
- Repair: what P0-B already authorizes. **GOVERNED** (system map §7.3)

**R-14 (Ch4 B7).** `utils/canonical_records.py:115-123`.
- What is wrong: five metadata names have no producer.
- Repair: P0-A, either classify them dead or give each a producer. **GOVERNED** (system map §7.1)

### 2.4 Step 4, adaptive meta-optimizer (Ch5)

**R-15 (Ch5 B1).** `adaptive_meta_optimizer.py:765-771`.
- What is wrong: the architecture score uses only the option's own size, so `[256,128,64]` is always chosen.
- Repair: make the choice depend on a named input, or state that the option list fixes it. **OPEN**

**R-16 (Ch5 B2).** `:345-373`, `:648`.
- What is wrong: entropy is computed and logged, but never used.
- Repair: feed it into the derivation, or document it as a log line only. **OPEN**

**R-17 (Ch5 B3).** `:415-454`.
- What is wrong: `get_regime_change_frequency` is never called.
- Repair: call it, or state that it is unused. **OPEN**

**R-18 (Ch5 B4).** `:471`, `:699`.
- What is wrong: nothing writes `reinforcement_feedback.json`, and its unfilled 0.05 weight stays in the denominator.
- Repair: name one producer, or state that the source is unconnected and drop the weight. **OPEN**

**R-19 (Ch5 B5 = Ch6 B7).** `meta_prediction_optimizer_anti_overfit.py` CLI `:3040-3105`.
- What is wrong: Step 5 never reads Step 4's output. The only coupling is a freshness check (`agents/watcher_agent.py:803-834` @a15b524).
- Repair: make Step 5 read the fields, or state that Step 4's output is an ordering precondition only. **OPEN**

### 2.5 Step 5, anti-overfit training (Ch6)

**R-20 (Ch6 B1).** `meta_prediction_optimizer_anti_overfit.py:2782-2797`.
- What is wrong: `test_mae=0`, `overfit_ratio=1.0` and `p_value=0.01` are hardcoded, and the test split is never scored.
- Repair: score the saved model on the test split and compute the overfit metrics from real errors. **OPEN**

**R-21 (Ch6 B2).** `mpo:230-259`.
- What is wrong: compare-models child runs don't receive `--k-folds`, `--output-dir`, `--enable-vmap` or `--persistent-workers`. 15 of 15 children ran 5 folds while the parent was given 20.
- Repair: forward every parameter, and show it in the child's log. **OPEN**

**R-22 (Ch6 B3).** `mpo:1813` vs `train_single_trial.py:522`, `:601`.
- What is wrong: the final NN ignores Optuna's chosen architecture and patience. The sidecar says 84-45-35-32; the checkpoint is `[256,128,64]`.
- Repair: rebuild the training config from the best trial, and prove the checkpoint matches the sidecar. **OPEN**

**R-23 (Ch6 B4).** `mpo:2497-2502` vs `:1105-1106`; `tst:381-382`.
- What is wrong: CatBoost's sampled `iterations` and `depth` are never used.
- Repair: use one set of key names, and record the parameters actually used in the sidecar. **OPEN**

**R-24 (Ch6 B5).** `prediction_generator.py:350-353`; `models/model_factory.py:334`.
- What is wrong: the schema hash is never validated, and missing features are filled with 0.0 (`pg:798`).
- Repair: compare the runtime schema with the sidecar, and fail on mismatch. **OPEN**

**R-25 (Ch6 B6).** `mpo:1440-1441`.
- What is wrong: `--study-name` and `--storage` are stored and never read.
- Repair: honour the flags, or remove them. **OPEN**

**R-26 (Ch6 B8).** `mpo:3074`, `:476-479`.
- What is wrong: the holdout offset is never recorded, and `holdout_window` is the CLI default of 1000.
- Repair: record the train length, holdout length and offset from the files actually used. **OPEN**

### 2.6 Step 6, prediction generator (Ch7)

**R-27 (Ch7 B1).** `survivor_scorer.py:136`; stub writer `window_optimizer_integration_final.py:2945-2953`.
- What is wrong: Step 6's survivor input is a count-only stub.
  - Dual-sieve mode raises `ValueError` (`int("survivor_count")`).
  - The other modes produce an empty pool and exit 0.
- Repair: feed Step 6 the population Step 3 scored, and make a stub or empty input fail loudly. **OPEN**

**R-28 (Ch7 B2).** `prediction_generator.py:375`, `:827-833`.
- What is wrong: a missing checkpoint falls back to `np.mean(X)` and reports `status: success`. The documented exit 2 is never returned.
- Repair: exit non-zero with a named reason, or govern the fallback and label its output as not model-based. **OPEN**

**R-29 (Ch7 B3).** `prediction_generator.py:849`.
- What is wrong: the pool is one value per top survivor, with no tally, weight or duplicate collapse.
- Repair: implement S179 §2.5. **GOVERNED** (`TB_RULING_S179_IMPLEMENTATION_AUTH.md` §2.5)

### 2.7 Agent framework (Ch10)

**R-30 (Ch10 B1).** `agents/registry/parameter_registry.py:263-600` (Step 4 entry `:437`).
- What is wrong:
  - 20 of 29 flags do not exist in the scripts they name.
  - Two script keys name files that never existed.
  - The Step-4 modes are wrong.
  - It feeds the LLM prompt (`agents/parameters/parameter_context.py:177-195`).
- Repair: make each entry match its script, or drop entries for scripts that do not exist. **OPEN** (dormant under `file_exists`)

### 2.8 Feature importance (Ch11)

**R-31 (Ch11 B1).** Readers: `feature_visualizer.py:69-72`, `web_dashboard.py:2043-2045`, `feature_importance_ai_interpreter.py:209-210`.
- What is wrong: global feature importance has no producer. It was removed at `e0b55a1` and `249052b`, with no governance record.
- Repair: name one producer keyed by the sidecar's 89 names, or state that these are standalone tools. **OPEN**

**R-32 (Ch11 B2).** `feature_importance.py:261-295`.
- What is wrong: global SHAP is not implemented, and no tracked `*.py` imports `shap`.
- Repair: implement it, or state which methods are actually supported. **OPEN**

**R-33 (Ch11 B3).** Integrated gradients are not implemented anywhere.
- Repair: implement them, or document plain saliency. **OPEN**

**R-34 (Ch11 B4).** `feature_importance.py:95-119` (and `:601-633`, `:679-705`); `feature_drift_tracker.py:197-221`.
- What is wrong: stale category lists. Global weight is always 0, and 45 of 89 names fall into "other".
- Repair: derive the lists from the sidecar or from the Step 3 producer. **OPEN**

### 2.9 WATCHER (Ch12)

At `a15b524`. Three items were repaired at `b6cb789` (report `docs/CLAUDE_CODE_REPORT_WATCHER_SAFETY_REPAIRS.md`).

**R-35 (Ch12 B1).** `agents/watcher_agent.py:946`.
- What is wrong: the file-exists check overrides `success:False`.
- **REPAIRED@b6cb789** (was GOVERNED, fact file §2.34/§2.51 #9)

**R-36 (Ch12 B2).** `agents/watcher_registry_hooks.py:151`.
- What is wrong: the fingerprint registry hooks are never called. The only importer is `agents/registry_inspector.py:30`.
- Repair: wire the pre- and post-run hooks with a fingerprint source, or have the owner decide the chapter should stop claiming them. **OPEN**

**R-37 (Ch12 B3).** `agents/safety/kill_switch.py:60` vs `agents/watcher_dispatch.py:126`, `:296`, `:386`.
- What is wrong: two halt files.
- **REPAIRED@b6cb789**

**R-38 (Ch12 B4).** `agents/watcher_dispatch.py:598-617`.
- What is wrong: request validation (age, priority, cooldown, load) is missing.
- Repair: implement it, or formally drop it. **OPEN**

**R-39 (Ch12 B5).** `agents/watcher_agent.py:1452-1454`.
- What is wrong: RETRY adjustments are only logged.
- Repair: pass bounded adjustments into the re-run, or document RETRY as a plain re-run. **OPEN** (dormant)

**R-40 (Ch12 B6).** `agents/watcher_agent.py:3836-3845`.
- What is wrong: `--dry-run` with `--run-pipeline` did nothing.
- **REPAIRED@b6cb789** (now refused)

### 2.10 Chapter 13 live feedback loop (Ch13)

**R-41 (Ch13 B1).** `chapter_13_diagnostics.py:114`.
- What is wrong: Chapter 13 reads a prediction artifact that nobody writes.
- Repair: one prediction artifact and one schema, shared by Step 6 and Chapter 13. **OPEN** (the Step 6 side is governed via `c6e3989`)

**R-42 (Ch13 B2).** `agent_manifests/full_scoring.json:120`.
- What is wrong: the loop's history and the pipeline dataset are disjoint.
- Repair: one dataset authority, the pointer manifest. **OPEN**

**R-43 (Ch13 B3).** `chapter_13_triggers.py:269`.
- What is wrong: a "20-draw" hit-rate trigger that is a one-draw miss detector.
- Repair: windowed Hit@K. **GOVERNED** (`TB_RULING_S176_WATCHER_KPI.md:55-116`; S179)

**R-44 (Ch13 B4).** `chapter_13_diagnostics.py:552`, `:559`.
- What is wrong: the decay and churn definitions.
- Repair: define them over identities and windows, and consume or drop the LLM threshold. **OPEN**

**R-45 (Ch13 B5).** `chapter_13_triggers.py:603`.
- What is wrong: the Chapter 13 approve commands do not reach the WATCHER executor.
- Repair: route them there, or stop advertising them. **OPEN**

**R-46 (Ch13 B6).** `agents/watcher_agent.py:3774` @a15b524.
- What is wrong: it runs the range `min..max` instead of the step list.
- Repair: execute the list. **OPEN** (`cf9edaa` changed learning-loop failure handling; whether it touches this is **[UNVERIFIED]**)

**R-47 (Ch13 B7).** `chapter_13_triggers.py:384`.
- What is wrong: a retrain executed by WATCHER never updates the cooldown or N-draw state.
- Repair: the executor that runs the retrain records it. **OPEN**

**R-48 (Ch13 B8).** `chapter_13_diagnostics.py:396`.
- What is wrong: no per-survivor prediction or hit tracking.
- Repair: add it, plus a consumer. **OPEN**

**R-49 (Ch13 B9).** `chapter_13_orchestrator.py:241`.
- What is wrong: the divergence detectors do not exist.
- Repair: implement them, or strike the section. **OPEN**

**R-50 (Ch13 B10).** `chapter_13_triggers.py:762`.
- What is wrong: nothing produces selfplay requests.
- Repair: a governed producer, behind the S179 gate. **OPEN**

**R-51 (Ch13 B11).** `chapter_13_acceptance.py:240`.
- What is wrong: promotion drops transforms, lineage and fingerprint.
- Repair: carry them through. **OPEN**

**R-52 (Ch13 B12).** `chapter_13_acceptance.py:780`.
- What is wrong: no shadow real-draw validation before promotion.
- Repair: add it. **OPEN**

**R-53 (Ch13 B13).** `inner_episode_trainer.py:148`.
- What is wrong: the selfplay objective and its gate are inconsistent with `holdout_quality`.
- Repair: one objective. **OPEN**

**R-54 (Ch13 B14).** `selfplay_orchestrator.py:606-607`.
- What is wrong: there is no outer-episode interface, and episodes have identical inputs.
- Repair: an interface on both sides, and something that varies between episodes. **OPEN**

**R-55 (Ch13 B15 = Ch14 B13).** `chapter_13_orchestrator.py:326`.
- What is wrong: the attribution blockers. The gate input is missing, as are the ranked-predictions producer, the sidecar key, the NN loader and the scaler.
- Repair: P0-C in full. **GOVERNED** (`TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md:89-101`)

**R-56 (Ch13 B16).** `chapter_13_orchestrator.py:412`.
- What is wrong: the `auto_executed_test_mode` outcome executes nothing; 120 cycles carry the label.
- Repair: execute, or rename the outcome. **OPEN**

**R-57 (Ch13 B17).** `chapter_13_acceptance.py:388`.
- What is wrong: the frozen-parameter set is hand-kept.
- Repair: derive it from the manifests. **OPEN**

**R-58 (Ch13 B18).** `chapter_13_orchestrator.py:510`.
- What is wrong: it grades a draw without checking that a prediction exists for it.
- Repair: check that the artifact exists. **OPEN**

**R-59 (Ch13 B19).** `chapter_13_diagnostics.py:52`, `:184-190`, `:448-464`.
- What is wrong: it reads a root-level sidecar and top-level hash keys, so `schema_hash_match` defaults to True.
- Repair: read `models/reinforcement/best_model.meta.json` → `feature_schema.feature_schema_hash`, and report "unknown" when no hash is available. **OPEN**

### 2.11 Chapter 14 training diagnostics (Ch14)

Abbreviations: `thc` = `training_health_check.py`, `td` = `training_diagnostics.py`, `tst` = `train_single_trial.py`, `mpo` = `meta_prediction_optimizer_anti_overfit.py`, `c13` = `chapter_13_orchestrator.py`.

**R-60 (Ch14 B1).** `thc:392` reads `best_round`; `td:239` writes `best_val_round`.
- Effect: 107 of 107 checks are critical, and the key decides the grade in 26 of 64 real reports.
- Repair: read the producer's key, and prove that a healthy report grades `ok`. **OPEN**

**R-61 (Ch14 B2).** `thc:373` vs `td:595-598`.
- What is wrong: the gradient-spread key differs between reader and producer.
- Repair: agree one key. **OPEN**

**R-62 (Ch14 B3).** `mpo:1832`, `:2761`, `:2587`.
- What is wrong: final tree models get no diagnostics.
- Repair: give the tree final-model path a capture. **OPEN**

**R-63 (Ch14 B4).** `tst:126-147`; `td:683-690`.
- What is wrong: XGB/LGB importance and per-round loss are never captured.
- Repair: capture them from the Booster objects actually built. **OPEN**

**R-64 (Ch14 B5).** `td:481`, `tst:147`.
- What is wrong: features are keyed `f{i}`/`feature_{i}`.
- Repair: pass the sidecar's 89 names into the trial worker. **OPEN**

**R-65 (Ch14 B6).** `thc:160-166` vs `mpo:311`, `:325`.
- What is wrong: the compare summary's filename and key differ between reader and writer.
- Repair: one name and one key. **OPEN**

**R-66 (Ch14 B7).** `diagnostics_llm_analyzer.py:183-191`.
- What is wrong: the LLM prompt reads old-schema keys.
- Repair: build it from the schema-1.1.0 keys. **OPEN**

**R-67 (Ch14 B8).** `thc:72`.
- What is wrong: `skip_duration_hours` is read nowhere.
- Repair: add an expiry, or state that skips last until reset. **OPEN**

**R-68 (Ch14 B9).** The §7.5 bridge functions are undefined; `watcher_failures.jsonl` and the Strategy Advisor bridge do not exist.
- Repair: one bridge artifact, with one writer and one reader. **OPEN**

**R-69 (Ch14 B10).** `td:834`; `thc:562`.
- What is wrong: the FIFO pruner is never called (111 files on the host).
- Repair: call it from the writer that runs. **OPEN**

**R-70 (Ch14 B11).** `c13:827-833`.
- What is wrong: `tier_comparison.json` is never produced.
- **GOVERNED** (P0-C)

**R-71 (Ch14 B12).** `selfplay_orchestrator.py:845`.
- What is wrong: it calls `LearningTelemetry.record_event`, which does not exist.
- Repair: add the method, or use an existing recorder. **OPEN**

**R-72 (Ch14 B14).** `c13:546-557`.
- What is wrong: the regression gate's fields have no producer.
- Repair: align what is emitted with what the gate tests. **OPEN**

**R-73 (Ch14 B15).** `c13:842`.
- What is wrong: `StrategyAdvisor.request_diagnostics_analysis` is undefined.
- Repair: call `diagnostics_llm_analyzer.py:308`. **OPEN**

**R-74 (Ch14 B16).** `thc:331-430`.
- What is wrong: `unused_feature_pct` is never evaluated.
- Repair: compute and evaluate it. **OPEN**

**R-75 (Ch14 B17).** `test_phase_8_soak.py:137-196`.
- What is wrong: tests 8.5 and 8.6 assert literals.
- Repair: tests that exercise production code. **OPEN**

**R-76 (Ch14 B18).** `td:429-431` vs `tst:515`.
- What is wrong: NN per-feature gradients are never captured; 0 of 75 real reports carry them.
- Repair: capture input gradients through a path that produces them, and show one real report that has them. **OPEN**

**Totals: 76 items.**

| Status | Count |
|---|---|
| OPEN | 67 (includes R-41, whose Step 6 side is governed) |
| GOVERNED | 6 (R-13, R-14, R-29, R-43, R-55, R-70) |
| REPAIRED@b6cb789 | 3 (R-35, R-37, R-40) |

---

## 3. Owner decision list: SWITCHED-OFF-BY-RULING, deduplicated (30)

Each item gives the ruling or approved spec that holds it, and what lifting it would enable. No item is recommended for or against. Under the 2026-11-11 authority ruling, every one is Michael's call.

Merged duplicates (6):
- Ch3 S1 = Ch12 S2.
- Ch9 S-4 = Ch12 S1.
- Ch4 S2 = Ch14 S5.
- Ch6 S2 is covered by Ch14 S5 and S6.
- Ch14 S2 = Ch13 S6.
- Ch14 S4 = Ch13 S5.

### 3.1 RANGE-MINER and cluster

**D-01 `use_range_miner` defaults to `false`** (`agent_manifests/window_optimizer.json:269`). (Ch12 S1 = Ch9 S-4)
- Held by: `PROPOSAL_S172_RANGE_MINER_v1_4_4.md:436`, `:503`, Phase 7 soak not run.
- Lifting it would enable: WATCHER Step 1 on the certifying engine by default.
- Note: it must flip together with `use_persistent_workers: false`. Today a default run reaches the PWC hybrid quarantine.

**D-02 Steps 2–6 through WATCHER on miner output.** (Ch12 S2 = Ch3 S1)
- Held by: `TEAM_ALPHA_PHASE7_LAUNCH_NOTICE.md` §1 (D3.0-B) and `gate12_launch.sh:73-76`.
- Lifting it would enable: a full 0→6 run. It also needs the converter path fixed. The certified run's `-nt` check (`run_scorer_meta_optimizer.sh:144`) always fires and then fails at `convert_survivors_to_binary.py:112`.

**D-03 `n_parallel` = 1 on the miner path.** (Ch12 S3)
- Held by: the D6.2 certification.
- Lifting it would enable: concurrent Optuna partitions, after a checkpoint design.

**D-04 Legacy fused reverse dispatch closed.** (Ch9 S-1)
- Held by: `S172_WINDOW_ANCHOR_BRIEF_I.md:312`; `TB_RULING_WINDOW_ANCHOR_BRIEF_I_CODE_REVIEW.md:70`.
- Lifting it would enable: `reverse_sieve_filter.py` dispatch again.

**D-05 PWC hybrid passes quarantined.** (Ch9 S-2)
- Held by: commit `8a55a68`; no ruling file was found.
- Lifting it would enable: PWC variable-skip passes, on kernels that ignore `skip_min`/`skip_max`.

**D-06 PWC/ZMQ have no certifying authority.** (Ch9 S-3)
- Held by: Beta, 2026-07-31, recorded in the fact file `:186-187` and Ch2 §10.
- Lifting it would enable: production use or comparator use.

**D-07 `generator_phase` pinned to 0.** (Ch9 S-5)
- Held by: TB 2026-08-21, cited at `range_miner_worker.py:699`.
- Lifting it would enable: DEP-ABI-V2, nonzero device-side phase.

**D-08 Five families uncovered on RANGE-MINER** (mt19937, philox4x32, sfc64, xorshift64, xoshiro256pp). (Ch8 S1)
- Held by: `PROPOSAL_S172_RANGE_MINER_v1_4_4.md` §11.I `:586-588`, §5.3.
- Lifting it would enable: the certifying engine sieving those substrates under the 22-array contract.

**D-09 Registry `offset` not yet `window_anchor`.** (Ch10 S1)
- Held by: `S172_WINDOW_ANCHOR_BRIEF_I.md:360-361`, `:675` (Brief II).
- Lifting it would enable: a resolver-delegated anchor maximum in the LLM registry.

### 3.2 Steps 3 to 6

**D-10 Holdout continuation exact only at skip 0.** (Ch4 S1)
- Held by: `BACKLOG.md` §8.1, "No fix is authorised".
- Lifting it would enable: holdout computed at the positions the sieve tested, for survivors with skip > 0.

**D-11 Step 4 reads Step 1's flat file, and its 60% "primary" source is constants.** (Ch5 S1)
- Held by: TB Q4 "preserve the drift", `PROPOSAL_S172_RANGE_MINER_v1_4_4.md:304`, `:723`, `:561-567`.
- Lifting it would enable: Step 4 output that follows what Step 1 actually found.

**D-12 Training target is `holdout_quality`, not `holdout_hits`.** (Ch6 S1)
- Held by: `S111_IMPLEMENTATION_PLAN_FINAL.md:6`, `:445`, `:481`.
- Lifting it would enable: training on the exact holdout hit rate again.
- Open item O1 (Ch6): the S111 burn-in versus the window-anchor ruling's continuation wording.

**D-13 `--output-dir` ignored by Step 6.** (Ch7 S1)
- Held by: `c6e3989` / `pg:902`; no ruling file was found.
- Lifting it would enable: Step 6 writing where it is told. WATCHER's success check would have to move with it.

### 3.3 Chapter 13 autonomy

**D-14 Retrain without a human.** (Ch13 S1)
- Held by: chapter §20.3; `PIPELINE_BEHAVIOUR_MODEL.md:954`; `TB_RULING_S179_IMPLEMENTATION_AUTH.md:525-532`.
- Lifting it would enable: unattended 3→6 or 1→6 runs. No execute path exists yet (R-56).

**D-15 KPI triggers audit-only.** (Ch13 S2)
- Held by: `TB_RULING_S176:795-821`; `S179:513-547`.
- Lifting it would enable: triggers requesting retrains.

**D-16 Applying LLM proposals.** (Ch13 S3)
- Held by: `CHAIN_C_TRUTHFULNESS_HOTFIX.md:104-106`.
- Lifting it would enable: accepted proposals changing parameters. This needs an apply function.

**D-17 Transform updates (Phase 9B.3).** (Ch13 S4)
- Held by: `PHASE_9B2_INTEGRATION_SPEC.md:5`, `:393-409`.
- Lifting it would enable: `propose_transform_update` returning changes.

**D-18 Acting on attribution / attribution-driven autonomy.** (Ch13 S5 = Ch14 S4)
- Held by: tracker v3_9 `:232-238`; system map `:222-228`, `:277-278`.
- Lifting it would enable: root-cause analysis feeding triggers, selfplay priors, or search.

**D-19 Activating attribution piecemeal / a real attribution run.** (Ch13 S6 = Ch14 S2)
- Held by: system map `:272`, `:275`, `:99`.
- Lifting it would enable: per-seed attribution of real predictions, once P0-C is done.

**D-20 Selfplay seeing live outcomes or promoting.** (Ch13 S7)
- Held by: `CONTRACT_SELFPLAY_CHAPTER13_AUTHORITY_v1_0.md:4-6`, `:106-123`, `:188-193`.
- Lifting it would enable: scoring selfplay on graded draws, or promotion without a human.

**D-21 Sieve thresholds in the LLM path.** (Ch13 S8)
- Held by: `TODO_SELFPLAY_AND_LLM_AUTONOMY.md:92-131`.
- Lifting it would enable: proposing and applying `forward_threshold`/`reverse_threshold`.

**D-22 Advisor-chosen Step-1 sampler.** (Ch13 S9)
- Held by: `BOUNDED_PHASE_6.md:98`, `:121`; `RUN4_ROUTING_PATCH_BRIEF_ROUTE_A.md:84`.
- Lifting it would enable: the advisor selecting the sampler.

### 3.4 Chapter 14 diagnostics

**D-23 Step-scoped param filter, so RETRY repeats the failed params.** (Ch14 S1)
- Held by: Chain C hotfix `:33-44`; `PIPELINE_BEHAVIOUR_MODEL.md:993-996`.
- Lifting it would enable: a RETRY that differs from the failed run.

**D-24 RCA observe-only.** (Ch14 S3)
- Held by: `S85.md:85`, `:95`.
- Lifting it would enable: `regime_shift` raising a retrain trigger.

**D-25 `forward_matches`/`reverse_matches` held out of the features.** (Ch14 S5 = Ch4 S2 = Ch6 S2 part)
- Held by: system map `:165-171`, `:276`.
- Lifting it would enable: per-seed sieve match quality reaching the model.

**D-26 Skip-shape placeholders stay zero; RANGE-MINER is capped at 22 arrays.** (Ch14 S6 = Ch6 S2 part)
- Held by: system map `:199-200`, `:268`.
- Lifting it would enable: per-seed skip shape reaching the model.

**D-27 No diagnostics on Optuna folds.** (Ch14 S7)
- Held by: S94 `:59`.
- Lifting it would enable: per-fold NN health during search.

**D-28 Selfplay trend check observe-only.** (Ch14 S8)
- Held by: S83 `:14`, `:46`.
- Lifting it would enable: a degrading trend stopping or adjusting selfplay.

**D-29 Dashboard / TensorBoard deferred.** (Ch14 S9)
- Held by: `CHAPTER_14_IMPLEMENTATION_PROPOSAL_S69.md:374-382`.
- Lifting it would enable: human-readable loss and importance views.

**D-30 S111 guardrails G1–G6 plus the `actual_mean`/`actual_std` exclusion.** (Ch14 S10, corrected in Addendum 2; also Ch11 G5)
- Held by: **the owner's decision**, "Hold off on S112 patch" (`SESSION_CHANGELOG_20260225_S111.md:84`), on a patch TB approved (`:80`). Whether the approval covers the exclusion is [UNVERIFIED].
- Lifting it would enable: two history-constant columns leaving training, plus the approved preflight and sidecar guardrails. It needs a schema-hash change and a retrain.

---

## 4. Updates needed elsewhere, not made in this pass

These were listed by the chapter subagents and checked by the main session only where noted. **None was made.**

### `docs/BACKLOG.md` §1
- Add the OPEN items of §2.
- Remove Chapters 3, 4, 5, 6, 7, 8, 10 and 11 from "unaudited/unscrutinised".

### `docs/PROJECT_FILE_CATALOG.md`
- **§7 gap 6:** Chapters 3–14 are audited.
- **`definitions.json`:** it is gitignored (`.gitignore:41`) and read by no code. It is stale: 62 features, `holdout_hits`, `ranked_survivors.json`, and a fatal Step-6 hash check that nothing implements.
- **Rows `:284`–`:290`, and gap 2:** Chapter 4 now describes `run_step3_full_scoring.sh`.
- **Gap 8:** the S179 implementation did not land.
- **Grammars:** there are two grammar directories, `grammars/` and `agent_grammars/`.

### Skill / fact file (`docs/TFM_PROJECT_FACTS_SKILL.md` v27.1)
- **Stale statements:**
  - "62 features" should be 91 extracted / 89 trained.
  - "WATCHER v1.8.0" is the manifest version; the program is v2.0.0.
  - "Supervised model · Steps 4–5" is wrong: Step 4 trains no model.
  - `:200-202`: the "enabled scraper unit" is disabled on VM 101, and its runner never existed.
  - `:1118-1122`: there is no per-seed skip and no vote weighting.
- **Things to add:**
  - The pipeline-level null results in §0.
  - The skip registry is a tracked runtime file, and NN skipping is inactive.
  - S111/S121 attributed the NN criticals to y-normalisation, while R-60 is the actual cause.
  - `nvidia-gpu-policy.service` fails on the one-GPU VM.
  - WATCHER defaults reach the PWC quarantine.
  - The `323b6fa` reverts (§7).

### `CLAUDE.md`
Record only; the brief forbids editing it.
- **§3:** "rig endpoint selectable … NOT done" is stale. `rig_profiles_config.json`, `execution_set.py` and `--rig-profile` exist.
- **§6:** it shows Phase 3 as next, but Phases 0–6 are done and Gate 12 passed. It names `EXPECTED_NPZ_KEYS`, which is not a symbol; the contract is `utils/canonical_arrays.py` `CANONICAL_ARRAY_CONTRACT`. "One-retry-then-fail-trial" applies to hybrid phases 3/4 only.
- **§7:** confirmed accurate.

### `docs/PIPELINE_BEHAVIOUR_MODEL.md`
- The "Chapter N is UNAUDITED / AUDITED and NOT CORRECTED" lines for Chapters 3–7 are stale.
- `:513-521`: the Step 4 flags and the fallback `{100,500,2000}` are wrong.
- I-2 cites `prediction.json` v1.6.0, but the file says 1.5.0.

### Frozen proposals: record only
- `PROPOSAL_S172_RANGE_MINER_v1_4_4.md:249`, `:320`: "runtime tolerates summary stub" is false (R-27).
- `PROPOSAL_S172_RANGE_MINER_v1_4_4.md:301`: the fallback wording.
- `PROPOSAL_Addendum_H_Feature_Importance_Integration.md`: "IMPLEMENTED" is false (R-31).

### Code and manifest text: for separate briefs
- **Seed-recovery phrasing in live prompts:** `agents/prompts/global_mission.py:6` and `feature_importance_ai_interpreter.py:260`.
- **Stale wording in manifests and comments:**
  - `watcher_policies.json:13`.
  - `agent_manifests/full_scoring.json:3`, "applies optimized scoring parameters" (R-12).
  - `agent_manifests/reinforcement.json:17`, `:43` and `mpo:620`, which still say `holdout_hits`.
  - `agent_manifests/window_optimizer.json:172` vs `:261`, a PWC self-contradiction; `:215`; `:336`.
  - `window_optimizer.py:1548-1551`.
  - The `scorer_trial_worker.py` docstrings.
  - `trse_step0.py:5`.
- **Legacy callers** passing the removed `--survivor-data`.
- **Host:** `daily3scraper.service` has a dead `ExecStart`, and `nvidia-gpu-policy.service` fails. Both are ops territory.

### Other documents
- `docs/WATCHER_POLICIES_REFERENCE.md:105`, "auto-executes" (R-56).
- `docs/SESSION_CHANGELOG_*S100.md:62-67`: the 20 folds never reached the children (R-21).
- `CHAPTER_14_TRAINING_DIAGNOSTICS.md:164` and `CHAPTER_3_SCORER_META_OPTIMIZER.md:1325`: both claim a SHAP method that does not exist (R-32).
- `utils/prng_encoding.py` cites "TB-approved Option C" with `java_lcg_hybrid=1`, against spec §5.4 `:386`. No ruling file was found.
- `docs/CHAPTER_13_LIVE_FEEDBACK_LOOP_v1_1.md` is a near-duplicate of Chapter 13.

---

## 5. §5 host questions and answers

All checks were read-only, on VM 101.

### Chapter 13: is there a host-side bridge that writes `prediction_pool.json` or raises `new_draw.flag`?
**No.**
- **systemd:** the only project unit, `daily3scraper.service`, is disabled. Its `ExecStart` runner `run_daily3scraper.py` has never existed, in any commit or anywhere on the filesystem. The scraper writes `daily3.json` only.
- **cron:** `crontab -l` shows none.
- **Writers:** there is no writer of `prediction_pool.json` in the tracked, untracked or ignored tree, under `~/`, or in history (`git log --all -S`).
- **`new_draw.flag`:** raised only by the test tools `synthetic_draw_injector.py:349`, `:408` and `draw_ingestion_daemon.py:265`.
- **UNAVAILABLE:** `.127`, the rigs, and root/other-user crontabs.
- Source: `CHAPTER_13_AUDIT.md` A.7.

### Chapter 14: which key do real `training_diagnostics` artifacts carry, `best_round` or `best_val_round`?
**`best_val_round`.**
- All 64 complete producer reports in `diagnostics_outputs/history/` carry it, and so does `neural_net_diagnostics.json` (2026-03-06).
- The only files with `best_round` are the health check's own self-test fixture (`thc:783-800`).

### Chapter 14: is `neural_net` in the skip registry?
**It has an entry, but it is not being skipped.**
- `consecutive_critical` is 1; the threshold is 3 (`mpo:174`).
- `skip_active` is read by no code.
- Every past drop below 3 was a hand edit (S90, S111, S121).

### Chapter 12: what does `use_range_miner` default to?
**`false`**, at `agent_manifests/window_optimizer.json:269`. This is recorded, not changed (D-01).

### All chapters: what is in `agent_manifests/definitions.json`?
- It exists: 6,635 B, schema 1.1.0, dated 2026-01-01/02. It is gitignored at `.gitignore:41` and was never tracked.
- It contains Step 5/6 sidecar contracts, the fingerprint protocol and a recovery workflow. It has no cluster, Step-2/3/4, registry or diagnostics configuration.
- **No `*.py` or `*.sh` reads it.**

---

## 6. Tokens per batch

These are subagent totals from task notifications. Main-session tokens are not reported by this session.

| Batch | Chapters | Subagent tokens |
|---|---|---|
| 1 | 9, 12, 13, 14 | ≈ 1.16 M (325k / 320k / 243k / 271k) |
| 2 | 3, 4, 5, 6, plus the Ch3 post-review fix | ≈ 1.19 M (300k incl. fix / 295k / 233k / 361k) |
| 3 | 7, 8, 10, 11 | ≈ 1.09 M (274k / 245k / 264k / 306k) |
| **Σ** | | **≈ 3.44 M** |

---

## 7. Lead: `323b6fa` "Doc sync 20260209_1704" reverted five chapters

This was a read-only check. **Nothing was restored.**

- The commit touched 24 files: +18,767 / −609. It copied an older snapshot over newer chapter text.
- That is the same defect class that destroyed Chapter 2 (`248e48c`, Chapter 2 §0.1). Its commit message describes none of it.
- For each chapter, the table gives the commit whose content was reverted.
- "Missing at HEAD" means non-blank removed lines that do not occur verbatim (stripped) at HEAD. The main session then searched each chapter at HEAD for the key terms. Script: the session scratchpad's `check_323b6fa.py`.

| Chapter | Removed by `323b6fa` | Content it reverted | Back at HEAD? |
|---|---|---|---|
| **Ch9** | 79 lines, all of §6 "GPU Process Isolation (Multi-Model Training)", sub-sections 6.1–6.5 | `709058c` (Session 72, 2026-02-08, one day earlier) | **Still missing.** 53 of 59 non-blank lines are absent. The terms "GPU Process Isolation", "-9999", `CUDA_INITIALIZED`, "DEFERRED to subprocesses" and `DESIGN_INVARIANT_GPU_ISOLATION` have 0 hits. A one-line pointer to `subprocess_trial_coordinator` was added this pass (`:137`). |
| **Ch12** | 81 lines, §6.4 "Step 5 Manifest: Multi-Model Comparison (Session 72)". The section appeared twice: once in a fenced form, once in plain form. | `709058c` | **Still missing.** 45 of 60 non-blank lines are absent. `compare_models` / `compare-models` has 0 hits in the chapter. |
| **Ch6** | 42 lines, §5.4 "CRITICAL: GPU Isolation Design Invariant" (mandatory invariant, implementation snippet, verification output, key-files table) | `709058c` | **Mostly missing.** 27 of 32 non-blank lines are absent. A pointer to `docs/DESIGN_INVARIANT_GPU_ISOLATION.md` and a `SubprocessTrialCoordinator` anchor were added this pass (`:609`, `:612`). The invariant text itself is gone. |
| **Ch3** | 4 lines: the NPZ v3.0 size row (57.9 MB / 733 KB) and the 2026-01-23 note that "NPZ v3.0 preserves all 22 metadata fields" | `2a1969b` (2026-01-23) | **Missing verbatim, superseded in substance.** The pre-v3.0 figures (258 MB / 0.6 MB) came back and are now marked UNVERIFIED (`:1155`, `:1165`). The chapter now has 8 references to the 22-array contract. The specific v3.0 sizes are not restored. |
| **Ch11** | 6 lines: the DeepSeek-R1-14B + Claude backup LLM update (`localhost:8080`) | `49e8ad4` (2026-01-20) | **Missing verbatim, superseded in substance.** The Qwen2.5-Math text it restored is still in the original sections (`:133`, `:966`, `:969`, `:1012`). The corrections this pass (`:148`, `:948-953`) state that the code uses DeepSeek on 8080, and name `323b6fa` as the revert. |

### The underlying documents still exist and still match the code
- `docs/DESIGN_INVARIANT_GPU_ISOLATION.md` is present (`709058c`).
- `meta_prediction_optimizer_anti_overfit.py:372` still has `CUDA_INITIALIZED = False  # Deferred`.
- `agent_manifests/reinforcement.json:176` still has `"compare_models": true`.

So the reverted Ch6, Ch9 and Ch12 sections describe live code that the chapters no longer document.

### Other chapter-named and session files the same commit rewrote
These were not analysed line by line, and should be checked the same way:
- `docs/CHAPTER_13_IMPLEMENTATION_PROGRESS_v3_0.md` went from v3.1.0 back to v3.0.0, removing the Soak A/B status table that `d80ec5e` had added. Those results also appear in later trackers: "Soak A" occurs in `_v3_1` (2 hits) and `_v3_9` (3 hits).
- `docs/SESSION_CHANGELOG_20260125.md`: −84 lines.
- `docs/SESSION_CHANGELOG_20260208_S73.md`: −161 lines.
- `docs/TODO_PHASE7_WATCHER_INTEGRATION_REVISED_v3.md`: −101 lines.

### Also in the same commit
It added copies of code files under `docs/`: `meta_prediction_optimizer_anti_overfit.py`, `train_single_trial.py`, `training_diagnostics.py`, `training_health_check.py`, `bundle_factory.py` and others. Those are stale duplicate source files; their status is the existing owner ruling and is not re-proposed here.

### What a restoration brief would have to do
Not done here.
- Re-add Ch9 §6, Ch12 §6.4 and Ch6 §5.4 from `git show 709058c:`, through the in-place, Edit-only, visible-trail rules.
- Re-verify each against code at the current HEAD.
- Check the four non-chapter files above.

---

## 8. Pass sentinel and VIR-1…6

| Chapter | Sentinel |
|---|---|
| 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14 | PASS |
| 9 | **INCOMPLETE** |

**Why Chapter 9 is INCOMPLETE:** 8 of its claims need rig access, which the brief forbade. They are TFLOPS/VRAM totals, the §8.4 benchmark and GFXOFF, PWC throughput, the S131/S146 records, the dashboard IP, and the rig-6600c deficit. Its other blocker, the main-session spot-check, has been done.

**Pass sentinel: `INCOMPLETE`.** It becomes `PASS` if Michael accepts Chapter 9's rig-dependent claims as UNAVAILABLE-by-scope.

**VIR declaration:**
- **VIR-1, execution proof.** Every audit lists the files it opened. The main session spot-checked anchors in every chapter: Ch3 10, Ch4 9, Ch5 6, Ch6 9, Ch7 7, Ch8 8, Ch9 9, Ch10 8, Ch11 7, Ch12 8, Ch13 7, Ch14 7. Every one matched; there were two off-by-one anchors, both corrected (Ch9 B-1 `:17`→`:18`, Ch13 B6 `:3773`→`:3774`).
- **VIR-2, clean control.** Chapter 2 is untouched; `git diff --quiet a15b524 -- docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` is empty. `tests/test_chapter2_content_gate.py` ran at the start of the session and after each batch: 12/12, SENTINEL PASS.
- **VIR-3, fault-injection control.** Not applicable to a documentation pass, and none was invented.
- **VIR-4, completion sentinel.** Per chapter, as in the table above. The pass sentinel is `INCOMPLETE`.
- **VIR-5, unavailable observers.** The rigs, `.127`, ser8, the private remote, root crontabs and `/tmp` halt files are UNAVAILABLE, not clean. `TEAM_BETA_REVIEW_kfolds_S100.docx` was not read, so R-21's "no governance record" covers readable files only.
- **VIR-6, scope.** What each chapter says, against what the code at `a15b524` does. Not whether the design is right. Searched surfaces, the governance trail and the chapters searched are listed in every audit's §8.

**Process deviations recorded:**
- **Ch14, batch 1:** one `sed -i` edit and one Python-replace edit. Both were verified lossless. This led to the Edit-only rule from batch 2 onward.
- **Ch8, batch 3:** piped the Chapter 2 gate through `tail` (CLAUDE.md rule 6). It was a short test, and nothing was lost.
- **Ch11:** its footer still reads "End of Chapter 14", which is original text.
- **Concurrent sessions:** other sessions changed the working tree during the pass (`b6cb789`, `cf9edaa`, the prod-shape scoping work). No subagent touched those files.

---

## 9. Gate (§8)

Stop. This session commits nothing.

Files this session wrote after `040697f`:
```
git add docs/CHAPTER_CORRECTIONS_REPORT_20261111.md docs/SESSION_CHANGELOG_20261111_CHAPTER_CORRECTIONS.md
```

The working tree also holds files from other sessions: `tests/gate_s172_prod_shape.py` (modified), `tests/test_gate_prod_shape_scoping.py`, `docs/CLAUDE_CODE_REPORT_PROD_SHAPE_RUN_SCOPING.md` and `data/`. They are not part of this pass and are not in the line above.

Dual-push: `git push origin main && git push public main`.
