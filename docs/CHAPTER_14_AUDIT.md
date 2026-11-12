# CHAPTER 14 AUDIT — Training Diagnostics vs. source

**Repo:** `/tmp/repos/prng_cluster_public` (public mirror, commit `a15b524`, single-commit history)
**Date:** 2026-11-11. Read-only audit. Every anchor below was read in this session.
**Audited:** `docs/CHAPTER_14_TRAINING_DIAGNOSTICS.md` (3,199 L), `docs/CHAPTER_14_HEADER_PATCH.md`, `docs/CHAPTER_14_IMPLEMENTATION_PROPOSAL_S69.md`, and the Chapter 14 status claims in `docs/CHAPTER_13_IMPLEMENTATION_PROGRESS_v3_9.md`.
**Framing held throughout:** TFM is functional mimicry of PRNG surface output. Nothing here assumes a "true seed" or a single stream.

Abbreviations: `mpo` = `meta_prediction_optimizer_anti_overfit.py`; `tst` = `train_single_trial.py`; `td` = `training_diagnostics.py`; `thc` = `training_health_check.py`; `psa` = `per_survivor_attribution.py`; `c13` = `chapter_13_orchestrator.py`; `wa` = `agents/watcher_agent.py`; `CH14` = the chapter.

---

## 1. Verdict

```
ACCURATE                 22 claims   (the modules exist and the NN path runs end to end under WATCHER)
BROKEN                   17 claims   (5 already recorded in whole or part, 12 with no governance record found)
SWITCHED-OFF-BY-RULING   10 items    (9 ruled/approved holds + 1 proposal still awaiting a ruling)
DESCRIPTION-ONLY         18 items
UNVERIFIED                5 items    (need torch, host state, or gitignored artifacts)
```

Short version: the diagnostics *producer* for the neural net is real and reachable; almost every *consumer* reads a different JSON schema than the producer writes, tree models emit no diagnostics on the production path, and the only automatic effect diagnostics have on the pipeline is a retry loop that re-runs Step 5 unchanged and a skip registry that removes the neural net from model comparison.

The counts are of the claims I checked, not of every sentence in 3,199 lines; the chapter's long sample-code listings (§8 says which) were read by heading and grep. BROKEN, SWITCHED-OFF and DESCRIPTION-ONLY are itemised in §4–§6, UNVERIFIED in §8. The 22 accurate claims:

| # | Claim | Anchor |
|---|---|---|
| 1 | `training_diagnostics.py` exists with a base class, factory, NN and tree subclasses, JSON save | `td:87`, `:123`, `:288`, `:321`, `:607` |
| 2 | NN forward and full-backward hooks on every `nn.Linear` | `td:366-373` |
| 3 | Activation mean/std and gradient norm captured per layer | `td:404-427` |
| 4 | Diagnostics are best-effort and never fail Step 5 | `tst:119-158`, `:610-618`, `:738-755`; `thc:80-120` |
| 5 | Absent diagnostics → PROCEED | `thc:126-135` |
| 6 | `--enable-diagnostics` on Step 5 and on the trial worker | `mpo:3080`; `tst:864` |
| 7 | NN hooks run inside the live epoch loop, reachable from WATCHER Step 5 | `tst:607-660`; `mpo:2587` |
| 8 | Detail snapshot every 5 rounds by default | `td:100`, `:447` |
| 9 | Output at `diagnostics_outputs/training_diagnostics.json` | `tst:101-107` |
| 10 | WATCHER reads diagnostics after Step 5 finishes, before Step 6; no mid-training abort | `wa:2861-2868` |
| 11 | Policy block in `watcher_policies.json` with the chapter's thresholds | `watcher_policies.json` `training_diagnostics` (values match `CH14:1147-1180`) |
| 12 | `check_training_health()` maps severity to PROCEED / PROCEED_WITH_NOTE / RETRY / SKIP_MODEL | `thc:433-464`; `wa:2262-2309` |
| 13 | Consecutive-critical registry, threshold 3, reset on success | `thc:467-535`; `wa:2284-2285` |
| 14 | Each health check archived to `history/` | `thc:539-569` |
| 15 | At most 2 retries, read from policy | `wa:2535-2545`, `:2871` |
| 16 | GBNF grammar and Pydantic schema for LLM analysis | `grammars/diagnostics_analysis.gbnf`; `diagnostics_analysis_schema.py:63-100` |
| 17 | LLM proposals validated against a whitelist with bounds | `wa:2311-2357` |
| 18 | `per_survivor_attribution.py`: four backends, dispatcher, tier comparison, `grad_x_input` default | `psa:49`, `:134`, `:184`, `:231`, `:289`, `:323`; default `:53` |
| 19 | Selfplay diagnostics history capped at 20 | `selfplay_orchestrator.py:523` |
| 20 | `post_draw_root_cause_analysis()` implements the chapter's hit/miss top-3 divergence rule | `c13:690-762` |
| 21 | Appendix: target is `holdout_quality`; NN `y` is normalised | `mpo:1481`; `tst:502-510` |
| 22 | Appendix: Battery Tier 1A adds 23 columns with the listed names | `survivor_scorer.py:760-766` |

---

## 2. The seven questions

### Q1. Tasks 8.1–8.7 — does the code path exist, and is it reachable from a production entry point?

The tracker says "All Tasks 8.1-8.7 validated" (`CHAPTER_13_IMPLEMENTATION_PROGRESS_v3_9.md:12`, `:72`). The chapter's own checklist says 8.1–8.3 ✅, 8.4 🔧, 8.5–8.7 ⬜ (`CH14:3068-3074`). The two documents disagree with each other before either is compared with code.

| Task | Code exists? | Reachable in production? | Anchors |
|---|---|---|---|
| **8.1** episode diagnostics in `inner_episode_trainer.py` | **Yes.** `train_single_fold` fits with `eval_set`, builds a `TrainingDiagnostics`, appends the report to a module-level side channel; `InnerEpisodeTrainer.train` aggregates folds. | **Only if `enable_diagnostics` is set true in the selfplay config. Nothing shipped sets it.** Default `False`; no CLI flag; WATCHER's selfplay spawn passes none; tracked `configs/selfplay_config.json` has no such key. | `inner_episode_trainer.py:271-293`, `:315-376`, `:549-598`; `selfplay_orchestrator.py:213`, `:776-778`, `:1101-1197` (no flag); `agents/watcher_dispatch.py:141-147`; `configs/selfplay_config.json` (10 keys, none diagnostic) |
| **8.2** aggregation in `selfplay_orchestrator.py` | **Yes.** `EpisodeResult.episode_diagnostics`, history list capped at 20. | Same condition as 8.1. | `selfplay_orchestrator.py:268`, `:502`, `:514-524`, `:818` |
| **8.3** `_check_episode_training_trend()` | **Yes** — in `selfplay_orchestrator.py`, not `chapter_13_orchestrator.py` as the tracker says (`…PROGRESS_v3_9.md:81`). Marked `OBSERVE-ONLY` in its docstring. | Same condition as 8.1. **Two parts do nothing even when reached:** (a) the return value is discarded by the caller; (b) the telemetry call uses `self.telemetry.record_event`, and `LearningTelemetry` has no such method, so the `AttributeError` is swallowed and the `training_quality_degrading` event is never recorded. | `selfplay_orchestrator.py:826-867` (docstring `:829`), call `:524`, telemetry `:843-853`; `modules/learning_telemetry.py` method list `:299`, `:376`, `:420` (no `record_event`, no `__getattr__`) |
| **8.4** `post_draw_root_cause_analysis()` | **Yes**, 6 methods, invoked from `run_cycle`. | **No.** Governed status: *implemented, invoked, unreachable, unconsumed* (`TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md:105-107`). Confirmed at this commit — see the six stops below. | `c13:321-347`, `:652-810` |
| **8.5** "run 10 selfplay episodes, verify diagnostics collected" | A harness exists. **Its synthetic mode does not call production code**: the 8.5 test builds a dict literal and asserts on its own keys. The file imports nothing from `inner_episode_trainer` or `selfplay_orchestrator`. | Not a production path. | `test_phase_8_soak.py:137-177`; imports `:37-46`, only project import is `c13` at `:378` |
| **8.6** "inject declining best_round_ratio, verify warning logged" | Same harness. The test filters a literal list against a test-local constant `TREND_WARNING_THRESHOLD = 0.3`; `_check_episode_training_trend` is never called (the production rule is "three strictly declining and last < 0.2", `selfplay_orchestrator.py:840`). | Not a production path. | `test_phase_8_soak.py:54`, `:179-196` |
| **8.7** "simulate hit-rate drop, verify RCA runs" | Real-mode harness calls the real Chapter 13 methods against files on disk. Validation used hand-built inputs (`SESSION_CHANGELOG_20260215_S87_FINAL.md:10-26`), and the archived outputs are synthetic (`S172_ATTRIBUTION_AND_FEATURE_TRACE_REPORT.md:555-584`). | Not a production path; proves the methods execute when handed inputs production never produces. | `test_phase_8_soak.py:367-652`; `test_task_8_4.py` |

**Where 8.4 stops (in call order):**

1. **Gate fields have no producer.** `_detect_hit_regression` opens only on a `summary_flags` entry containing both "hit_rate" and "drop", or on top-level `hit_at_20 < previous_hit_at_20` (`c13:546-557`). The diagnostics producer emits 19 possible flags, none matching (`chapter_13_diagnostics.py:585-668`), and its output dict has neither key (`:855-870`). `previous_hit_at_20` appears nowhere else in tracked `*.py`. Recorded as status, not new: "has never fired in production" (`SESSION_CHANGELOG_20260217_S95.md:178-180`).
2. **No writer for `predictions/ranked_predictions.json`.** Reader `c13:876-937`; the only other reference is the test harness (`test_phase_8_soak.py:60`). System-map Blocker 1 (`…SYSTEM_MAP…:91`).
3. **Wrong sidecar key.** `c13:582` reads `meta.get("feature_names")`; Step 5 writes it under `feature_schema` (`mpo:1293`, `:2903`, `:2972`). Blocker 2 (`:93`).
4. **NN loader.** `c13:596-598` does `torch.load(...)` then `.eval()`; Step 5 saves a checkpoint dict (`tst:708`, `:725`). Blocker 3 (`:95`).
5. **NN attribution omits the training scaler.** `psa:98-101` feeds the raw vector; training standardises (`tst:484-492`). Blocker 4 (`:97`).
6. **Missing method.** On `regime_shift` the code calls `StrategyAdvisor.request_diagnostics_analysis(...)` (`c13:842`); no such method is defined anywhere in tracked `*.py` (the only hit for the name is that call). The exception is caught at `c13:854-855`. I found no governance record of this one.

### Q2. The training target

- **Chapter body:** never names the target. The motivating numbers are a December 2025 MSE table (`CH14:37-44`). The only statement is the appended S111 note: "holdout_quality (composite score) replaced holdout_hits" (`CH14:3163-3172`). The chapter does not give the 50/30/20 composition.
- **Code:** Step 5 trains on `holdout_quality`. `mpo:1479-1483` (`target_field='holdout_quality'`, exclude list `['score','confidence','holdout_hits','holdout_quality']`); loader default `mpo:612`; the value is a top-level record key written by Step 3 (`full_scoring_worker.py:423`), found by the loader's `elif target_field in item` branch (`mpo:657`).
- **Composite confirmed:** `holdout_quality.py:99-122` — 0.50 × CRT match (0.40 `residue_1000_match_rate`, 0.20 each `lane_agreement_8`, `lane_agreement_125`, `lane_consistency`) + 0.30 × coherence (mod 1000/8/125) + 0.20 × `temporal_stability_mean`, clamped to [0,1] (`:125`).
- **Stale `holdout_hits` wording still in code/config:** loader docstring `mpo:620`; `agent_manifests/reinforcement.json:17` and `:43`.
- **Selfplay trains on something else.** `selfplay_orchestrator.py:933` prefers `holdout_hits`, and when a record has a `features` dict it uses that dict whole (`:924-925`) — all 91 keys including `score` and `confidence`. So episode diagnostics (8.1–8.3), if enabled, describe a 91-feature model against `holdout_hits`, not the Step-5 model (89 features, `holdout_quality`). The target half is recorded as open in the fact file §2.1.
- **Units differ between families.** The NN trains on z-scored `y` (`tst:502-510`), so its diagnostic losses are in normalised units; trees train on raw `y`. The chapter cites this at "line 499" (`CH14:3176`); it is now `:502-510`.
- **Adjacent, not a chapter claim:** the S111 autocorrelation diagnostic that was meant to check whether `holdout_quality` is real persistence or scorer self-consistency has **no caller**. Defined at `mpo:59-73`; the S111 patch only inserted the call if a regex matched `survivors = json.load(...)` (`apply_s111_holdout_quality_redesign.py:273-289`), and no such line exists in `mpo`. `diagnostics_outputs/holdout_feature_autocorr.json` is therefore never written by a pipeline run.

### Q3. The feature contract

**91 extracted / 89 trained — confirmed by counting the batch producer.**

| block | count | anchor |
|---|---|---|
| match/statistics (incl. `score`, `confidence`) | 27 | `survivor_scorer.py:702-731` |
| residue, mods 8/125/1000 × 3 | 9 | `:624-667`, merged `:734` |
| battery Tier 1A | 23 | `:754-757`; names `:760-766` |
| metadata merge list | 18 | `:774-779` |
| `global_*` | 14 | `full_scoring_worker.py:347-349`, `:403-405` |
| **total** | **91** | minus `score`, `confidence` (`mpo:1482`) = **89** |

**Trained features that are constant or have no producer.** The fact file says six. I count **nine** that are constant for every survivor regardless of how many trials are in the dataset, before the run-globals and trial-level fields are considered:

| feature | why | anchor |
|---|---|---|
| `skip_mean`, `skip_std`, `skip_entropy`, `survivor_velocity`, `velocity_acceleration` | In the Step-3 merge list, but not among the 24 canonical record fields, so never in metadata; defaulted to 0.0. No assignment found in tracked `*.py` outside fixtures. | merge `survivor_scorer.py:776-778`; default `:784-791`; `utils/canonical_records.py:115-123` |
| `best_offset` | Hardcoded. | `survivor_scorer.py:707` (`torch.zeros`), sequential `:373` |
| `total_predictions` | `float(n)`, `n = len(lottery_history)` — same for every seed. | `:706`, `:560` |
| `actual_mean`, `actual_std` | Mean/std of the draw history, broadcast to every seed. | `:716-717`, `:608-609` |

`actual_mean`/`actual_std` are **already recorded**: S111 found `actual_mean` correlating r = +1.0000 with the target through float noise and recommended excluding both, "Pending TB Approval" (`S111_TEAM_BETA_BRIEFING.md:84-105`, `:152-157`; `SESSION_CHANGELOG_20260225_S111.md:178`). The exclude list at this commit still has four entries (`mpo:1482`), so both are still trained. I found no ruling answering that request and no record naming `total_predictions`.

Then, identical for every survivor **in a run**: the 14 `global_*` (`full_scoring_worker.py:348-349`). And identical for every survivor **of a trial**: the 13 NPZ-backed fields (Q5).

**In a single-generation survivor set, 36 of the 89 trained columns carry one value** (9 + 14 + 13), leaving 53 that vary by seed. The NN scaler maps a zero-variance column to scale 1.0 (`tst:488-490`), i.e. to exactly 0 after centring.

**No diagnostic checks for this.** The one policy bound that would — `unused_feature_pct`, "features with zero importance (trees) or zero gradient (NN)" — is declared in the chapter (`CH14:1169`), in `watcher_policies.json:159`, and in `thc:67`, and is evaluated nowhere: `_evaluate_metrics` (`thc:331-430`) has no branch for it and `td` computes no such metric.

### Q4. `forward_matches` / `reverse_matches`

**Confirmed.** The merge list at `survivor_scorer.py:774-779` names exactly 18 fields; neither is among them, nor are `window_size`, `offset`, `trial_number`, or the NPZ `score`. Both are produced per seed (`utils/canonical_records.py:248-251`, record fields `forward_match_rate`/`reverse_match_rate`) and described by the converter as "the surface fingerprint signals that ML uses to rank survivors" (`convert_survivors_to_binary.py:16-20`). The sequential fallback merges even less — six fields (`full_scoring_worker.py:453-454`).

Governed: system map §7.3 defect 3 and its ruling, "do NOT silently add them" (`…SYSTEM_MAP…:165-171`), authorization table `:276`.

**What the diagnostics therefore cannot see:**
- How well each seed matched the observed window in either direction. No importance list, gradient, SHAP value, tier comparison or LLM prompt can mention it, because the column is not in `X`.
- Whether sieve match quality relates to holdout quality at all. That question is not answerable from any Chapter 14 artifact.
- Any per-seed sieve evidence whatsoever: every sieve-derived column that *does* reach the model is a trial-level count (Q5). The per-seed information the model has is entirely Step 3's own regeneration of the seed's sequence (with `skip=0`, `survivor_scorer.py:575`, documented `CHAPTER_4_FULL_SCORING.md:523`) compared position-by-position with the training history.

### Q5. Trial-level values stamped on every seed

**All 13 NPZ-backed trained features are trial-level.** They are built once into `metadata_base` and copied into every record of the trial:

`skip_min`, `skip_max`, `skip_range`, `forward_count`, `reverse_count`, `bidirectional_count`, `intersection_count`, `intersection_ratio`, `forward_only_count`, `reverse_only_count`, `survivor_overlap_ratio`, `bidirectional_selectivity`, `intersection_weight`.

Anchors: miner path `utils/canonical_records.py:216-244` (dict) and `:246-257` (`record.update(metadata_base)` per seed); PWC path `window_optimizer_integration_final.py:1791-1813` (splatted per seed at `:1830`), whose own comment reads "Trial-level context (same for all seeds in this trial)" (`:1788`). In the canonical record only `seed`, `forward_match_rate`, `reverse_match_rate`, `score` are set per seed (`:248-253`) — and none of those four reaches the model.

**Do the diagnostics treat them as per-seed signal? Yes — by not distinguishing at all.** No file among `td`, `psa`, `thc`, `diagnostics_llm_analyzer.py` contains a feature namespace, a `global_` test, or a constant-column test (grep for `global_`, `trial`, `namespace`, `constant`, `context_features` returns only unrelated "CONSTANTS" headers).

- `psa` assigns an attribution share to every name positionally (`psa:117`, `:173`, `:220`, `:269`).
- Chapter 13 RCA takes each seed's top-3 attributed names and declares `regime_shift` when the hit and miss name-sets diverge by > 0.5 (`c13:707-715`, `:743-762`). A trial-level name in a top-3 is counted as evidence distinguishing seeds.
- `compare_pool_tiers` reports top-20 minus top-300 mean attribution per feature (`psa:399-428`) with the same blindness.
- NN `spread_ratio` is max/min gradient magnitude over all inputs (`td:486`), constants included.
- Because the attribution path omits the scaler (Blocker 4), `grad_x_input` (`psa:105-106`) multiplies each gradient by the **raw** value. Raw trial counts (`forward_count`), `total_predictions` and `actual_mean` are orders of magnitude larger than the per-seed rates, so if the path ever ran unscaled they would dominate the normalised shares on magnitude alone. This is a reading of the code, not an observed result.

The governance position exists and is not implemented: globals "must not be searchable as ordinary per-survivor filter thresholds", per-survivor attribution to one "is contextual model dependence, not evidence distinguishing survivors" (`…SYSTEM_MAP…:191-197`); "no diagnostic module may maintain a manually duplicated feature-name list" (`:208`).

**The chapter's own examples make the category error.** It presents `intersection_weight` (trial-level), `forward_count` (trial-level, "ranges 0-50K") and `skip_entropy` (no producer) as the features that explain why "THIS specific seed" ranks where it does: `CH14:49`, `:110-111`, `:432-433`, `:533-537`, `:2127-2128`, `:2141-2143`, `:2234`. The code's self-test fixtures repeat it (`td:1035`, `diagnostics_llm_analyzer.py:568`).

### Q6. RANGE-MINER, the 22-array contract, the window-anchor change

**None is mentioned.** Case-insensitive search of all three Chapter 14 documents for `range.miner`, `range_miner`, `S172`, `npz`, `22.array`, `window.anchor`, `window_anchor`, `generator.phase`, `PWC`, `forward_matches`, `reverse_matches`, `trial-level`: zero hits except the S111 `holdout_*` lines. The stale items are listed in §7.

### Q7. Does diagnostic output feed anything automatically?

**Two automatic effects exist. Both are driven by a defect.** Everything else stops at a file or a log line.

| # | Output | Producer | Artifact | Consumer | What it changes |
|---|---|---|---|---|---|
| 1 | NN per-epoch diagnostics | `tst:607-660`, saved `:738-746`; reached via WATCHER Step 5 → manifest `enable_diagnostics: true` (`agent_manifests/reinforcement.json:179`) → `mpo:3122` → per-model child `mpo:237-238` → final NN model `mpo:1824-1829` → flag at `mpo:2587` | `diagnostics_outputs/neural_net_diagnostics.json`, copied to `training_diagnostics.json` (`tst:101-107`) | `check_training_health()` (`thc:126-186`), called at `wa:2866` | → rows 4, 5 |
| 2 | Tree diagnostics | `tst:112-158` | `{model}_diagnostics.json` | — | **Not produced on the production path** (B3). |
| 3 | Per-run archive | `thc:539-569` | `diagnostics_outputs/history/{model}_{ts}.json` | `diagnostics_llm_analyzer.py:368-370`, last 5 by filename sort | Context in an LLM prompt only. Strategy Advisor does not read this directory — it reads `diagnostics_history/` (`parameter_advisor.py:210`, `:500`). Never pruned (B10). |
| 4 | Health action RETRY | `thc:433-515` → `wa:2262-2309` | in-memory | `wa:2870-2905` | **Re-runs Step 5, up to 2 times** (`wa:2535-2545`), after deleting the diagnostics file (`wa:2897-2901`). The modified params do not arrive (S1). |
| 5 | Consecutive-critical counter | `thc:498-503`; reset `thc:518-535` | `diagnostics_outputs/model_skip_registry.json` (a copy is tracked in the repo) | `mpo:151-181` | **Removes that model family from `--compare-models`** at ≥ 3. This is the one place diagnostics alter model selection. |
| 6 | LLM `DiagnosticsAnalysis` | `diagnostics_llm_analyzer.py:308-461`, called only on RETRY (`wa:2400-2432`) | `diagnostics_outputs/llm_proposals/*.json` (`:470-484`) | `wa:2484-2520` | Proposals validated, then dropped at the step boundary (S1). The archive has no reader. |
| 7 | Compare summary | `mpo:311`, `:325` | `compare_models_summary_{run_id}.json` | none — the health check looks for a different filename and key (B6) | Nothing. |
| 8 | Episode diagnostics | `inner_episode_trainer.py:549-598` | `EpisodeResult.episode_diagnostics` (memory) | `_check_episode_training_trend` (`selfplay_orchestrator.py:826`) | A `print`. Not passed to telemetry (`:581-592`); event call fails (B12). Off by default. |
| 9 | Root-cause analysis | `c13:652-810` | `diagnostics_outputs/history/root_cause_{ts}.json` (`c13:860-873`) | cycle-record summary `c13:349-355`; "Does NOT affect trigger evaluation" (`c13:321-323`) | Nothing. Unreachable anyway. |
| 10 | Tier comparison | `c13:827-833` only | `diagnostics_outputs/tier_comparison.json` | LLM prompt (`diagnostics_llm_analyzer.py:215-226`), `wa:2405` | Never produced, so readers always get `None`. |
| 11 | Autocorrelation | `mpo:59-73` | `holdout_feature_autocorr.json` | — | No caller. |

**Chapter 13 does not consume training diagnostics.** Its triggers read post-draw diagnostics (`chapter_13_triggers.py:174`, `:216`). The one reference to `training_diagnostics.json` is inside the unreachable regime-shift branch (`c13:837`).

**Model choice is not diagnostic-driven.** The compare-models winner is picked by an R²-like score read from each sidecar (`mpo:94-139`, `:288-296`).

**What rows 4 and 5 actually do at this commit** — reproduced by running `thc.check_training_health()` in a scratch directory against two reports written in the producer's real schema:

```
healthy NN report (best_val_round 90/100, spread 50000) -> RETRY critical
     ['WATCHER: Early stop ratio 0.00 (peaked at round 0/100) — severe overfitting']
empty tree report (no rounds)                           -> RETRY critical
     ['WATCHER: Early stop ratio 0.00 (peaked at round 0/1) — severe overfitting']
```

The health check reads `training_summary['best_round']` (`thc:392`); the producer writes `best_val_round` (`td:239`). The default is 0, the ratio is always 0.00, and 0.00 ≤ the critical bound 0.15 (`thc:398`). **Any diagnostics file that exists is graded critical.** The same run shows the gradient-spread cross-check silently not firing on a 50,000× value (B2).

Consequence traced through the code: with the tracked manifest defaults, only the NN writes the file → critical → RETRY ×2 → third check hits the registry threshold → `SKIP_MODEL` → next run drops `neural_net` from the comparison (`mpo:174-177`). No tree model writes diagnostics, so the stale NN file stays on disk and keeps being graded. The 24-hour expiry does not exist (B8); the counter resets only on a non-critical result (`wa:2284-2285`), which cannot occur while a file is present. The repo's history agrees with the trace: "'Early stop ratio 0.00' as automatic critical needs revisiting" (`SESSION_CHANGELOG_20260215_S92.md:188` — attributed there to thresholds, not to the key), "neural_net with 14 consecutive criticals" (`mpo:152-153`), and a tracked registry entry carrying a hand-written `reset_reason` (`diagnostics_outputs/model_skip_registry.json`). Current host behaviour is **[UNVERIFIED]** (§8).

---

## 3. What the network is actually fed

89 trained columns (`mpo:1482`; schema from the first record's sorted keys, `mpo:566-567`). Step 3 batch path is the producer unless stated.

| group | n | features | producer anchor |
|---|---|---|---|
| **A. Per-seed and informative** | **53** | | all from regenerating the seed's sequence (`skip=0`, `survivor_scorer.py:575`) and comparing with the whole training history |
| A1 match | 1 | `exact_matches` | `:592`, `:705` |
| A2 prediction/residual stats | 8 | `pred_mean`, `pred_std`, `pred_min`, `pred_max`, `residual_mean`, `residual_std`, `residual_abs_mean`, `residual_max_abs` | `:596-606`, `:708-715` |
| A3 CRT lanes | 3 | `lane_agreement_8`, `lane_agreement_125`, `lane_consistency` | `:612-614` |
| A4 digit agreement (S119) | 4 | `hundreds_digit_agreement`, `tens_digit_agreement`, `ones_digit_agreement`, `expected_digit_match_count` | `:618-621` |
| A5 temporal stability | 5 | `temporal_stability_{mean,std,min,max,trend}` | `:670-690` |
| A6 residue | 9 | `residue_{8,125,1000}_{match_rate,kl_divergence,coherence}` | `:624-667` |
| A7 battery Tier 1A | 23 | `batt_fft_*` (5), `batt_ac_*` (12), `batt_cs_*` (3), `batt_bf_*` (3) | `:754-757`; computed from the generated sequence only |
| **B. Trial-level, stamped on every seed of the trial** | **13** | `skip_min`, `skip_max`, `skip_range`, `forward_count`, `reverse_count`, `bidirectional_count`, `intersection_count`, `intersection_ratio`, `forward_only_count`, `reverse_only_count`, `survivor_overlap_ratio`, `bidirectional_selectivity`, `intersection_weight` | `utils/canonical_records.py:216-244`, stamped `:254`; merged `survivor_scorer.py:771-781` |
| **C. Run-level, identical for every survivor** | **17** | 14 × `global_*`; `total_predictions`, `actual_mean`, `actual_std` | `full_scoring_worker.py:347-349`, `:403-405`; `survivor_scorer.py:706`, `:716-717` |
| **D. Constant zero, no producer or hardcoded** | **6** | `skip_mean`, `skip_std`, `skip_entropy`, `survivor_velocity`, `velocity_acceleration`; `best_offset` | default `survivor_scorer.py:784-791`; `:707` |
| | **89** | | |

Notes on reading the table:
- Group A's `residue_1000_match_rate`, both lane agreements, `lane_consistency`, the three coherences and `temporal_stability_mean` are the training-window versions of exactly the eight quantities the target is composed from on the holdout window (`holdout_quality.py:99-115`). That is the design (S111), and it is why the autocorrelation check in Q2 was specified.
- Group B varies between rows only when the dataset merges several trials; each row then carries its winning trial's values (fact file §2.22). In a one-generation set it is constant.
- Not fed at all, though present in the NPZ: `forward_matches`, `reverse_matches`, NPZ `score`, `window_size`, `offset`, `trial_number`, `sessions`, `skip_mode`, `prng_type`.
- On GPU-batch failure the sequential fallback merges 6 metadata fields instead of 18 (`full_scoring_worker.py:453-454`), zeroing seven of Group B. Governed P0 (`…SYSTEM_MAP…:162-164`).

---

## 4. BROKEN

"Governed" = already recorded in the governance trail; listed as status, not as a discovery. "No record found" = searched `docs/TB_RULING_*`, `TB_RULING_REQUEST_*`, `PROPOSAL_*`, `TEAM_ALPHA_*`, the system map, fact file, behaviour model, `BACKLOG.md` and the S69–S121 changelogs.

| # | What the chapter/tracker says | Where the code stops | Governance | A repair would have to |
|---|---|---|---|---|
| B1 | Health check grades early-stop ratio from best/total rounds (`CH14:1160-1167`) | `thc:392` reads `best_round`; producer writes `best_val_round` (`td:239`). Every file → critical. Reproduced. | Symptom recorded, cause not (`…S92.md:188`) | Make the consumer read the key the producer writes, and prove a healthy report grades `ok`. |
| B2 | WATCHER cross-checks NN gradient spread against policy (`CH14:1150-1154`) | `thc:373` reads `model_specific.feature_gradient_spread`; producer writes `model_specific.gradient_health.gradient_spread_ratio` (`td:596-598`). Never evaluated. Reproduced. | No record found | Same: one agreed key, read where it is written. |
| B3 | Live introspection for all four families; checklist 3.3 ✅, tracker Phase 3 "VERIFIED end-to-end under WATCHER" | Final tree models train inline with no diagnostics hook: `mpo:1832` (Optuna path), `mpo:2761` (single-shot). `_cli_enable_diagnostics` is consumed only for the NN at `mpo:2587`. The older subprocess path reads a global nothing sets (`mpo:756`) and is unreachable from the CLI since the S88 rewrite (`mpo:3110-3125`). | No record found. Header patch notes "NN-first approach" (`CHAPTER_14_HEADER_PATCH.md:55-58`), a session note, not a ruling | Give the tree final-model path a diagnostics capture, or route tree final models through the path that has one. |
| B4 | XGBoost/LightGBM per-round loss + gain/split importance (`CH14:242-352`) | Even where `_emit_tree_diagnostics` runs, only CatBoost's rounds are read (`tst:126-141`); importance needs `feature_importances_` (`tst:144`), which the `lgb.train`/`xgb.train` Booster objects built at `tst:216` and `:309`, `:323` are not given. `XGBDiagnostics`/`LGBDiagnostics` are empty subclasses (`td:683-690`). Booster attribute behaviour not executed here (libraries absent). | No record found | Capture rounds and importance from the objects these functions actually build. |
| B5 | Importances and gradients keyed by feature name (`CH14:429-436`) | NN diagnostics constructed with no names (`tst:614`) → `feature_{i}` (`td:481`); trees keyed `f{i}` (`tst:147`). The subprocess receives `X` only. | No record found | Carry the sidecar's 89 names into the subprocess; system map §9 authority order applies. |
| B6 | Health check evaluates the compare-models winner (S93 Bug B fix; `thc:156-178`) | Looks for `compare_models_summary.json` with key `winner_model_type`/`winner` (`thc:160-166`); writer emits `compare_models_summary_{run_id}.json` with `winner_best_effort` (`mpo:311`, `:325`). Override never fires; the gate judges the NN's file whoever won. | No record found | Agree one filename and key between writer and reader. |
| B7 | LLM receives rounds, top features, NN layer health (`CH14:2119-2134`) | Prompt builder reads `training_rounds`, `feature_importance.values`, `nn_specific.*` (`diagnostics_llm_analyzer.py:176-188`); producer writes `training_summary`, `model_specific.*` (`td:207-213`). The LLM sees model type, severity and issue strings only. | No record found | Build the prompt from the schema-1.1.0 keys. |
| B8 | SKIP_MODEL lasts 24 h, tracked in `model_skip_state.json` (`CH14:1176-1180`, `:2816`) | `skip_duration_hours` is defined (`thc:72`, policy file) and read nowhere; no code writes `model_skip_state.json`. A skip persists until a non-critical check or a manual edit. | No record found | Implement an expiry, or state that skips are permanent until reset. |
| B9 | Diagnostics and LLM analysis feed the Strategy Advisor; incidents logged to `watcher_failures.jsonl` (`CH14:1706-1716`, `:2596`, `:2613-2618`) | `_record_incident`, `_record_model_skip`, `_request_diagnostics_llm`, `_update_strategy_advisor` are not defined in any tracked module (`_record_model_skip` exists only in the unimported `watcher_integration_snippet.py:158`). Advisor reads `diagnostics_history/`, not `diagnostics_outputs/history/` (`parameter_advisor.py:210`, `:500`). | Partly governed: "`strategy_recommendation.json` → WATCHER: no code reads the file" (fact file §2.13, `:788`) | Define the bridge artifact and give it one writer and one reader. |
| B10 | FIFO history pruning complete (tracker Phase 5, `…PROGRESS_v3_9.md:68`) | `_prune_history_fifo` is called only from `MultiModelDiagnostics.save` (`td:834`) and a CLI self-test (`td:1060`); nothing instantiates `MultiModelDiagnostics`. The live archive writer (`thc:562`) does not prune. | No record found | Call the pruner from the writer that actually runs. |
| B11 | Step 5 writes `tier_comparison.json` (`CH14:2568-2569`) | Step 5 never imports `psa`. Sole producer is `c13:827-833`, unreachable. | Governed (`S172_ATTRIBUTION…REPORT.md:426`, `:614`) | Part of P0-C (S2). |
| B12 | Degrading trend recorded as a WATCHER incident (`CH14:2407-2414`) | `selfplay_orchestrator.py:845` calls a method `LearningTelemetry` lacks; swallowed at `:852-853`. | No record found | Add the event method or use an existing recorder. |
| B13 | Chapter 13 runs RCA when Hit@20 drops (`CH14:2425-2558`, `:2606-2610`) | Four blockers, Q1 stops 2–5. | **Governed** (`…SYSTEM_MAP…:89-101`); repair order fixed at `:255-256` | P0-C in full; a one-line patch is ruled insufficient (`:272`). |
| B14 | Same path — the gate | Q1 stop 1: trigger fields have no producer. | Governed as status (`…S95.md:178-180`); the trace report classed the gate "IMPLEMENTED AND WIRED" (`S172_ATTRIBUTION…:626`) | Have the Chapter 13 diagnostics emit what the gate tests, or make the gate test what is emitted. |
| B15 | Same path — LLM on regime shift | Q1 stop 6: `StrategyAdvisor.request_diagnostics_analysis` undefined (`c13:842`). | No record found | Point the call at the function that exists (`diagnostics_llm_analyzer.py:308`). |
| B16 | `unused_feature_pct` bound (`CH14:1169-1173`) | Declared in three places, evaluated in none (`thc:67`; no branch in `thc:331-430`; no metric in `td`). | No record found | Compute the metric and evaluate it; it is the only bound that would surface Groups C and D. |
| B17 | Tasks 8.5 and 8.6 validated (tracker `:91-99`) | Synthetic tests assert on literals and never call production code (`test_phase_8_soak.py:137-196`). | No record found | Tests that import and exercise `InnerEpisodeTrainer` and `_check_episode_training_trend`. |

---

## 5. SWITCHED-OFF-BY-RULING

For each: the hold, where it is recorded, what lifting it would make possible, and what else would have to change. No recommendation is made.

**S1. Diagnostic and LLM proposals do not reach Step-5 parameters.**
- *Ruling:* "The step-scoped filter is a deliberate executable-interface boundary, not an oversight… The six proposals must not be wired through." / "This does not authorize adding the six fields to `reinforcement.json`." — quoted in `CLAUDE_CODE_INSTRUCTIONS_CHAIN_C_TRUTHFULNESS_HOTFIX.md:33-44`; upheld `PIPELINE_BEHAVIOUR_MODEL.md:993-996`; fact file `:789`. No standalone `TB_RULING_*` file for it exists in the clone.
- *Mechanism:* `wa:1813-1820` merges only keys in the manifest's `default_params`. The LLM whitelist (`wa:2336-2343`: `normalize_features`, `nn_activation`, `learning_rate`, `dropout`, `n_estimators`, `max_depth`) and the heuristic suggestions `dropout`, `normalize_features`, `use_leaky_relu` (`wa:2375-2386`) are all absent from `agent_manifests/reinforcement.json:173-184`. Only `model_type` passes, and `--compare-models` ignores it (`mpo:147`). Net: a retry is the same Step 5 again (continuing the Optuna study). The LLM log line was corrected (`wa:2441-2518`); the three heuristic `[WATCHER][RETRY]` lines still claim a change (`wa:2378`, `:2382`, `:2386`; `BACKLOG.md:96-100`).
- *If lifted:* a retry could differ from the run that failed.
- *Would also need:* the six names declared in the Step-5 manifest; a CLI route for each in `mpo` (today only `--dropout` exists, and normalisation/LeakyReLU are forced on for the NN, `mpo:245-246`); B1 fixed first, otherwise every run triggers adaptation; B7 fixed, otherwise the LLM proposes from issue strings alone.

**S2. Real per-survivor attribution run.**
- *Ruling:* "Real attribution run — **Blocked** until schema, scaler and prediction-parity gates pass"; "One-line Ch13 feature-name patch alone — Not sufficient; do not land in isolation" (`…SYSTEM_MAP…:272`, `:275`); Gate A-NORM (`:99`).
- *If lifted:* Chapter 13 could attribute real predictions to features per seed.
- *Would also need:* everything in P0-C (`:255-256`) — a producer for the ranked-prediction artifact with name-keyed features and schema hash, the canonical loader, the NN scaler, prediction parity — plus B14 and B15. Lifting the ruling alone changes nothing; the path does not execute.

**S3. Root-cause classification is observe-only.**
- *Ruling:* TB review round 2, "Recommended observe-only before granting trigger authority" (`SESSION_CHANGELOG_20260214_S85.md:85`, `:95`); tracker `:232-238`; system map §11 (`:222-228`). Code: `c13:321-323`.
- *If lifted:* a `regime_shift` classification could raise a retrain trigger.
- *Would also need:* S2; a trigger type for it in `chapter_13_triggers.py`; and the evidence conditions the ruling lists (multi-draw evidence, minimum cohort sizes, stability, disjoint rank bands) — a single Pick-3 draw often leaves a hit cohort of zero or one.

**S4. Attribution-driven autonomous parameter changes.**
- *Ruling:* "**Not authorized**" (`…SYSTEM_MAP…:277`); observe-only prior experiment "Authorized after P0 activation" (`:278`).
- *If lifted:* attribution could steer search or parameters without a human step.
- *Would also need:* S2; a consumer (none exists — `S172_ATTRIBUTION…:639`); signed contributions, since the engine keeps only absolute shares (`psa:105-108`; system map §5, `:121-125`); and a local/global split so Groups B–D are not read as seed evidence.

**S5. `forward_matches` / `reverse_matches` as model features.**
- *Ruling:* "do NOT silently add them… Open a separately governed feature-schema decision", Option A or B (`…SYSTEM_MAP…:167-171`, `:276`).
- *If lifted (Option A):* the model and every diagnostic would see per-seed sieve match quality for the first time.
- *Would also need:* a new feature-schema version; both names in the merge list and in the sequential fallback's list; regenerated Step-3 artifacts; all four families retrained; the leakage and redundancy analysis the ruling names; Step 6 and attribution accepting the new schema hash. The miner already emits both arrays.

**S6. Dead placeholders must stay zero under the current schema.**
- *Ruling:* "They must not begin receiving nonzero values under the existing 89-feature schema" (`…SYSTEM_MAP…:199-200`); three lifecycle options listed.
- *If lifted (option 3):* `skip_mean`, `skip_std`, `skip_entropy` could carry the per-seed skip shape the GPU already computes.
- *Would also need:* the host to stop discarding `skip_sequences` (fact file §2.2, `extract_survivor_records`); a carrier for them — "Expanding RANGE-MINER beyond 22 arrays — Not authorized" (`:268`) bears on that; a schema version bump and retrain. `survivor_velocity` and `velocity_acceleration` have no designed producer that I found.

**S7. No diagnostics on Optuna folds.**
- *Ruling:* "TB Trim #1: Diagnostics only on final model, not per fold (avoids 100-file explosion)" (`SESSION_CHANGELOG_20260216_S94.md:59`; code comment `mpo:1891-1892`).
- *If lifted:* per-fold training health for the NN during search.
- *Would also need:* per-fold output paths (the canonical file is a single path that each writer overwrites, `tst:106-107`) and a pruning policy that runs (B10).

**S8. Selfplay trend detection is observe-only.**
- *Recorded:* "Observe-only — no automatic intervention" (`SESSION_CHANGELOG_20260213_S83.md:14`, `:46`); docstring `selfplay_orchestrator.py:829`. A session decision; I found no separate TB ruling text.
- *If lifted:* a degrading trend could stop or adjust a selfplay run.
- *Would also need:* `enable_diagnostics` turned on in the selfplay config; B12; the caller using the return value (`:524`). Note the XGBoost and LightGBM branches record the validation loss as the training loss (`inner_episode_trainer.py:328`, `:339`), so their overfit gap is always 0 and only CatBoost folds can leave `ok`.

**S9. Web dashboard and TensorBoard.**
- *Recorded:* deferred in the approved-scope proposal (`CHAPTER_14_IMPLEMENTATION_PROPOSAL_S69.md:374-382`), header patch (`:63`), tracker "Web Dashboard — Future — Lower priority" (`…PROGRESS_v3_9.md:74`). A deferral, not a prohibition.
- *State:* no `/training` route (`web_dashboard.py` routes `:1943-2225`); no `tensorboard`/`SummaryWriter` string in any tracked `*.py`; no `--enable-tensorboard` flag.
- *If built:* human-readable loss curves and importances.
- *Would also need:* B3–B5, otherwise three of four families have nothing to plot and features are unnamed.

**S10. `actual_mean` / `actual_std` exclusion and guardrails G1–G6 — a request still awaiting a ruling.**
- *Recorded:* "Next Steps (Pending TB Approval)" (`S111_TEAM_BETA_BRIEFING.md:152-157`); guardrails `:115-135`. No answering ruling found in the trail. Under the 2026-11-11 authority change this is the owner's to decide.
- *If approved:* the two history-constant columns leave training; a preflight would flag near-constant and target-correlated columns.
- *Would also need:* a schema-hash change (the hash is over the name list, `mpo:587-588`), so retrain and Step-6 revalidation; a decision on `total_predictions` and Groups C/D, which G1/G3 would also flag.

Also a design decision, authored by Team Beta inside the chapter rather than issued as a ruling: **post-training read only, no mid-training abort** (`CH14:1099-1117`). The code matches it (`wa:2861-2866`).

---

## 6. DESCRIPTION-ONLY

| # | Chapter says | Code / current fact |
|---|---|---|
| D1 | 62 features throughout (`CH14:220`, `:495-496`, `:505`, `:563`, `:590`, `:624`, `:646-647`, `:973`, `:3187`); proposal says 47 (`…PROPOSAL_S69.md:112`, `:311`) | 89 trained / 91 extracted |
| D2 | NN hooks inserted in `reinforcement_engine.py` (`CH14:157`, `:2774`, `:3027`) | Production wiring is `tst:607-660`. `reinforcement_engine.py:697-712` also has it, on a path Step 5 does not use |
| D3 | Unified JSON schema 1.0.0 with `training_rounds`, `feature_importance.values`, `nn_specific` (`CH14:418-455`) | Schema 1.1.0 with `training_summary`, `model_specific`, `round_data_sample` (`td:68`, `:207-213`). The broken consumers (B2, B7) were written to the chapter's schema |
| D4 | Nine health-check functions added to `agents/watcher_agent.py` (`CH14:1708-1724`) | They live in `training_health_check.py`, imported at `wa:103-108` |
| D5 | `DiagnosticsBundle` in `bundle_factory.py` (`CH14:1730-1900`, `:2281-2283`) | `build_diagnostics_prompt` in `diagnostics_llm_analyzer.py:111`; `bundle_factory.py` has no diagnostics code |
| D6 | `overfit_ratio`: "Train loss / val loss" (`CH14:1160`) | val / train (`thc:415`; `watcher_policies.json`) |
| D7 | One `diagnostics` block in `reinforcement_engine_config.json` with `episode_diagnostics`, `tensorboard`, `nn_attribution_method`, `top_survivors_for_attribution` (`CH14:2786-2799`) | `ReinforcementConfig.diagnostics` defaults (`reinforcement_engine.py:272-276`) and a separate `SelfplayConfig.enable_diagnostics` (`selfplay_orchestrator.py:213`). No reader for the attribution keys |
| D8 | Header: "Phases 1, 3, 5, 6 Complete" (`CH14:6`) | Own checklist marks 2 and 7 ✅ and 8 partial; tracker says 8 complete; header patch says "Phases 1-3 Complete", "Phase 2 DEFERRED" (`CHAPTER_14_HEADER_PATCH.md:24`, `:49`). "Phase 5" means TensorBoard in the chapter and FIFO pruning in the patch (`:63-65`) and tracker |
| D9 | §13.1: Soak A/C ⬜, "No code changes until soak tests pass" (`CH14:2864-2867`) | §14 marks both ✅ at S76/S77 after phases 1, 3, 6 landed in S69–S73 |
| D10 | Tracker: `_check_episode_training_trend()` in `chapter_13_orchestrator.py` (`…PROGRESS_v3_9.md:81`) | `selfplay_orchestrator.py:826` |
| D11 | Example outputs driven by `intersection_weight`, `forward_count`, `skip_entropy` (lines in Q5) | Two trial-level, one with no producer |
| D12 | "Dead neuron % … Is ReLU killing neurons?" (`CH14:152`) | Chapter and code both measure exact zeros on the `nn.Linear` output (`td:368-372`, `:404-412`), which sits before BatchNorm and the activation (`models/wrappers/neural_net_wrapper.py:66-69`). Production forces LeakyReLU (`mpo:245-246`). Expected value ≈ 0 by construction — not observed here |
| D13 | Selfplay code shown as `train_episode_model`, `run_inner_episodes`, key `episode_diagnostics` (`CH14:2304`, `:2347`, `:2359`) | `train_single_fold` / `InnerEpisodeTrainer.train_best`, `SelfplayOrchestrator.run`, `enable_diagnostics` |
| D14 | "train_single_trial.py line 499" (`CH14:3176`) | `tst:502-510` |
| D15 | Checklist 3.1 ✅ lists `--enable-tensorboard` (`CH14:3026`) | Flag does not exist |
| D16 | One set of thresholds, in `watcher_policies.json` (`CH14:1147-1173`) | Two layers: producer `td:75-80` (dead 15/25, spread 1000/10000, gap ratio 0.3/0.5) and WATCHER policy (25/50, 100/1000, 1.3/1.5) |
| D17 | Per-sample attribution is "a unique capability no tree model has" (`CH14:68-70`, `:2747-2749`) | The chapter's own §4.3–4.5 and `psa:134-273` give trees per-sample attribution; corrected in the system map (`:237`) |
| D18 | Config default `"enabled": false`, "Default: diagnostics OFF" (`CH14:2790`; proposal `:181`) | Step-5 manifest default is `enable_diagnostics: true` (`agent_manifests/reinforcement.json:179`, set in S90) |

---

## 7. Stale because of RANGE-MINER, the 22-array contract, or the window-anchor change

The chapter mentions none of the three. What they make stale:

1. **Every empirical number in the chapter.** The December 2025 MSE table and "100,000x worse" (`CH14:37-46`), the worked LLM example (`:2119-2275`), and the S111 table — mod-8 ≈ 32 %, "24/62 base features 0 %", R² +0.0046 and +0.020538 (`:3165-3187`). All were measured on PWC-produced survivors, and all from the window the sieve could then reach (`data[0:150]`, fact file §2.21). The first also predates the `holdout_quality` target. The window anchor is now a separate quantity from generator phase (`utils/canonical_records.py:216-224`), so the population these numbers describe is not the population a current run produces.
2. **`forward_count` "ranges 0-50K" as the per-survivor scaling culprit** (`CH14:49`, `:537`, `:2234`). Under the canonical record it is `len(fwd_map)`, one value per trial (`utils/canonical_records.py:227`).
3. **`intersection_weight` as a per-seed driver** (`CH14:110`, `:432`, `:533`). Trial-level (`:236`).
4. **`skip_entropy` as a top feature** (`CH14:111`, `:433`, `:534`, `:2127`, `:2142`). The 22 arrays carry no skip-shape statistic; the column is zero.
5. **The implied survivor scale.** "95K survivors" (`CH14:37`) and S111's 49,882; the certified miner generation recorded in the fact file is 319 seeds (§2.22). Thresholds, `capture_every_n`, and the "2-8 minutes" runtime argument for post-training-only reads (`CH14:1113`) were sized to the old scale.
6. **Feature count and names** (D1). The 91/89 contract and the three namespaces were established by the S172 trace; the chapter's 62 cannot be reconciled with the 13/59/14/5 split.
7. **No statement of which arrays reach diagnostics.** The chapter has no place where a reader learns that 4 arrays carry per-seed information and none of the 4 is a model input. That is the single most relevant fact for interpreting any importance or attribution output.
8. **`offset`.** Nothing in the chapter is wrong about it, because the chapter never mentions it; but a reader will meet `best_offset` in the feature list. It is hardcoded 0 (`survivor_scorer.py:707`) and is unrelated to array 4, which after Brief I is the window anchor (`utils/canonical_records.py:216-224`).
9. **"Step 3 (26 GPUs)"-era assumptions** do not appear in the chapter; nothing to correct there.

---

## 8. Surfaces searched, and surfaces unavailable

**Searched, in the required order**
- *Governance:* `docs/TB_RULING_*` (24 files), `docs/TB_RULING_REQUEST_*`, `docs/TB_UPDATE_*`, `docs/PROPOSAL_*`, `docs/TEAM_ALPHA_*` — grep for `chapter 14`, `training_diagnostics`, `training_health`, `per_survivor_attribution`, `diagnostics`, `allowed_params`, `step-scoped`, `skip=0`, `actual_mean`; read in full: `TEAM_ALPHA_AUTONOMY_CONTROL_SURFACE_SUBMISSION.md`, `TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md`, `CLAUDE_CODE_INSTRUCTIONS_CHAIN_C_TRUTHFULNESS_HOTFIX.md` (§0–2), `S111_TEAM_BETA_BRIEFING.md`; read in part: `S172_ATTRIBUTION_AND_FEATURE_TRACE_REPORT.md`, `BACKLOG.md`, `PROJECT_FILE_CATALOG.md` (Chapter 14 rows).
- *Orientation:* fact file §0, §1.x, §2.1–2.4, §2.13, §2.20, §2.22, §2.53 (part); behaviour model §1.1, §5, §6, §7, §9, §12, §13.2.
- *Chapters:* Chapter 14 — §1–2, §3.1, §3.5, §4, §5.1, §7.1–7.2, §7.5, §8.7, §9, §10.3, §11–14 and the appendices read line by line; §3.2–3.4, §5.2–6, §7.3–7.4, §8.1–8.6, §10.2 (sample code) read by heading and targeted grep, not line by line. Header patch, S69 proposal and progress tracker read in full.
- *Changelogs:* S83, S85, S87, S88–S95, S111 by targeted grep.
- *Code (tracked `*.py`, excluding `apply_*`, `*.bak`):* the files listed in the brief plus `train_single_trial.py`, `training_health_check.py`, `diagnostics_llm_analyzer.py`, `diagnostics_analysis_schema.py`, `inner_episode_trainer.py`, `chapter_13_orchestrator.py`, `chapter_13_diagnostics.py`, `full_scoring_worker.py`, `utils/canonical_records.py`, `window_optimizer_integration_final.py`, `convert_survivors_to_binary.py`, `modules/learning_telemetry.py`, `parameter_advisor.py`, `agents/watcher_agent.py`, `agents/watcher_dispatch.py`, `reinforcement_engine.py`, `subprocess_trial_coordinator.py`, `web_dashboard.py`, `test_phase_8_soak.py`; tracked JSON `agent_manifests/reinforcement.json`, `watcher_policies.json`, `configs/selfplay_config.json`, `diagnostics_outputs/model_skip_registry.json`.
- *Patch corpus:* `apply_s111_holdout_quality_redesign.py`, `apply_s93_bug_a_b_c_fixes.py`, `apply_s93_bug_b1_fix.py` — read for intent only.
- *Executed:* one scratch-directory run of `training_health_check.check_training_health()` on two hand-written reports (Q7). Nothing else was run.

**Unavailable — and where it matters**
- **Git history.** The mirror has one commit. I cannot say when B1, B3 or B6 entered, or confirm that S88 is what disconnected tree diagnostics; that is inferred from the code and the S88/S93 patch scripts.
- **Gitignored files and host state.** `reinforcement_engine_config.json` is not in the clone, so the diagnostics block the chapter specifies is unread. The live selfplay config, the live skip registry, real `*_diagnostics.json`, `compare_models_summary*`, and `diagnostics_outputs/history/` on Zeus are invisible. A host-side file could turn on selfplay diagnostics or supply `compare_models_summary.json`; a clone cannot show it.
- **The private repo and ser8 archives.** Not searched. Any "no record found" in §4 is a statement about the surfaces listed above.
- **TB's primary review text for the attribution trace.** Cited through the system map, which records it as binding; no `TB_RULING_*` file carries it in this clone. The same holds for the Chain C ruling, quoted in the instructions file.
- **torch, xgboost, lightgbm, catboost, optuna** are not installed here.

**[UNVERIFIED] items and what would verify each**
1. **NN per-feature gradients are probably never captured in production.** The chapter specifies an extra forward/backward pass with `requires_grad` on the input (`CH14:213-219`); the code instead reads `grad_input[0]` from the first layer's backward hook (`td:429-431`) while the training input is a plain tensor (`tst:515`). In PyTorch that entry is `None` for an input that does not require grad, which would mean `feature_gradients`, `top_10` and `spread_ratio` never appear. *Verify:* attach `NNDiagnostics` to a two-layer net with a non-grad input and look for `feature_gradients` in `round_data`; or open any real `neural_net_diagnostics.json` on Zeus.
2. **`dead_pct` ≈ 0 on every layer** (D12). *Verify:* the same real file.
3. **Steady-state NN exclusion** (Q7). *Verify:* the live `model_skip_registry.json` and one WATCHER Step-5 log.
4. **The tracker's soak results** (30/30 cycles, "real SHAP attribution"). Historical runs on inputs production does not produce; not replayable here.
5. **Tree `Booster` objects lacking `feature_importances_`** (B4). *Verify:* run `_emit_tree_diagnostics` on a model from `lgb.train` / `xgb.train`.

**One process error to record.** My scratch run of `training_health_check` wrote a `__pycache__/` directory into the repo root (the `-I` flag suppressed the environment variable meant to prevent it). I removed it immediately; `git status --short --ignored` is empty and no tracked file was touched.

---

## Addendum — live-host re-verification, 2026-11-11

**Host:** VM 101 (`zeus-ubuntu`, `192.168.3.177`) as `michael`, working tree `/home/michael/distributed_prng_analysis`.
**HEAD:** `a15b5247336255801de8cf67d52653701b1e8c80` (`git rev-parse HEAD`) — the same commit the audit above was made from, so line anchors are expected to match; this pass checked that they do.
**Authority:** `docs/CLAUDE_CODE_INSTRUCTIONS_CHAPTER_CORRECTIONS_20261111.md` REV1, §2, §3.1 ("re-verify it instead"), §5.
**Scope:** read-only. No training, health check, pipeline step, service or rig was run or touched. Nothing above this heading was changed. The only executed code was `python3 -I` one-liners that open JSON files and count keys.

### A.0 Null results first

- **No file named `diagnostics_outputs/training_diagnostics.json` exists on the host now.** The canonical path the health check reads (`training_health_check.py:56`, `DIAGNOSTICS_PATH`) is absent. Under `thc:126-135` the next health check would return PROCEED / `absent` unless a new NN Step-5 run writes one. The audit's "the stale NN file stays on disk" is therefore **not the current state** — see A.2, correction C3.
- **`neural_net` is NOT currently in the skip state.** See A.1 Q2.
- **No xgboost or lightgbm diagnostics report exists anywhere on the host** (0 of 111 archived health checks; `find / -xdev -name '*_diagnostics.json'`). This supports B3; it does not prove it.
- **No `watcher_failures.jsonl`, `model_skip_state.json`, `state/` or `agent_state/` exists on the host** (`find / -xdev`, `ls`). This supports B8 and B9.
- **`reinforcement_engine_config.json` does not exist on the host** (`find . -name 'reinforcement_engine_config*'` → nothing). The diagnostics config block the chapter specifies (`CH14:2786-2799`) has no file; only the code defaults apply (D7).

### A.1 The two §5 host questions

**Q1. Does a real training-diagnostics artifact carry `best_round` or `best_val_round`?**
**Answer: `best_val_round`. No real producer artifact on the host carries `best_round`.**

- Code anchors, both confirmed this session: producer `training_diagnostics.py:239` writes `"best_val_round": best_val_idx` inside `_compute_training_summary`; consumer `training_health_check.py:392` reads `training_summary.get('best_round', 0)` in `_evaluate_metrics`. Both lines date from the same day: `best_round` read since `881ec3b` (2026-02-08, "Chapter 14 Phase 6"), `best_val_round` written since `51e74b7` (2026-02-08, "Chapter 14 Phase 1") (`git log -S`).
- Artifacts opened (all gitignored by `.gitignore:41` `*.json`, so invisible to the public mirror):

| artifact | mtime (host local) | `schema_version` | key in `training_summary` |
|---|---|---|---|
| `diagnostics_outputs/neural_net_diagnostics.json` (4,947 B) | 2026-03-06 21:32:40 -0800 | `1.1.0` | `best_val_round` = 1 (of `rounds_captured` 17) |
| `diagnostics_outputs/history/neural_net_20260308_030751.json` (newest archive) | 2026-03-07 19:07:51 -0800 | `1.1.0` | `best_val_round` = 1 |
| all 111 files in `diagnostics_outputs/history/` (104 `neural_net_*`, 4 `catboost_*`, 7 `root_cause_*`) | 2026-02-08 … 2026-03-07 | — | tabulated below |

| archived health checks (107, excluding the 7 `root_cause_*`) | n | `training_summary` key |
|---|---|---|
| producer schema 1.1.0, status `complete` (60 NN + 4 CatBoost) | 64 | `best_val_round` |
| producer schema 1.1.0, status `partial` (NN, empty summary) | 14 | none |
| no `schema_version`, `rounds_captured` 100, `best_round` 15, losses 0.01/0.02 | 2 | `best_round` — **this is the health check's own self-test fixture** (`training_health_check.py:783-800`), not producer output |
| no `schema_version`, hand-built (`metrics.dead_neuron_ratio` etc.) | 27 | none |

- **What the grader did with them:** all 107 archived checks have `watcher_severity: "critical"`. 105 carry `"WATCHER: Early stop ratio 0.00 …"`; the other 2 are the fixture, graded 0.15 (also critical). This is the audit's B1 observed on real host data rather than reproduced in a scratch directory.
- **Refinement of B1's consequence (computed from the artifacts with the rules in `thc:331-430`, not by running the health check):** if the consumer read `best_val_round`, the 64 real complete reports would split **38 still critical / 26 not critical**. Of the 38: 25 are already `critical` from the producer's own diagnosis (`thc:196-198` starts from it), and 13 have a real early-stop ratio ≤ 0.15. So the key mismatch decides the grade in **26 of 64** real reports, not all 64. The newest real report (above) would be critical either way: producer severity `critical` ("Severe overfitting (gap ratio: 4.18)") and val/train 5.18 ≥ 1.5.

**Q2. Is `neural_net` currently in the skip registry?**
**Answer: it has an entry, and it is NOT being skipped.**

- Registry location: `diagnostics_outputs/model_skip_registry.json` (code: `training_health_check.py:59` `SKIP_REGISTRY_PATH`; consumer `meta_prediction_optimizer_anti_overfit.py:153-177` in `_s88_run_compare_models`). It is the only file in `diagnostics_outputs/` that is tracked (`git ls-files diagnostics_outputs`), and the host copy is identical to HEAD (`git diff --stat HEAD` empty).
- Live content (mtime 2026-03-07 19:07:51 -0800): `{"neural_net": {"consecutive_critical": 1, "last_critical": "2026-03-08T03:07:51.608806+00:00", "consecutive_failures": 0, "skip_active": false, "reset_reason": "S121: Y-norm fix confirmed working, R2=+0.02"}}`.
- Exclusion rule: `consecutive_critical >= max_consecutive_critical` (`mpo:174`, filter `:178`); the threshold is 3 (`watcher_policies.json` `training_diagnostics.model_skip_rules.max_consecutive_critical`, read this session). 1 < 3 → `neural_net` stays in `--compare-models`.
- `skip_active`, `consecutive_failures` and `reset_reason` are read by no code (`/bin/grep -rn` over tracked `*.py`, excluding `apply_*` and `.bak`). `skip_active: false` therefore does not enforce anything; only the counter does.
- History (`git log -- diagnostics_outputs/model_skip_registry.json`, each version opened with `git show`): 4 at `fc1cf2b` (S91, 2026-02-15, `reset_reason` "S90: Fresh start…"); 3 at `c596649`, `4e340de`, `5ea338d`, `6082188` (S96A–S102); 4 at `ca975f8` (S109); 3 at `d860f6f` (S111); 0 at `b2a89d8` (S122, after the S121 hand reset); 1 at `888cf3e` (S123). **`neural_net` was at or above the exclusion threshold in every committed version from S91 to S111**, and each return below it was a hand edit (S90, S111 — `SESSION_CHANGELOG_20260225_S111.md:85-88`; S121 — `SESSION_CHANGELOG_20260307_S121.md:45`). This resolves audit UNVERIFIED #3.

### A.2 Re-verification of every claim group above

**Anchors.** Every code anchor in §1–§7 was re-opened at the live tree (≈150 line anchors across `td`, `tst`, `mpo`, `thc`, `wa`, `psa`, `c13`, `selfplay_orchestrator.py`, `inner_episode_trainer.py`, `diagnostics_llm_analyzer.py`, `modules/learning_telemetry.py`) and the governance anchors in the system map, S92/S94/S95/S111 changelogs, S111 briefing and progress tracker. All matched the stated content except C1 below.

| item | result | evidence (this session) |
|---|---|---|
| §1 accurate claims 1–22 | **22 confirmed** | anchors as listed; claim 9 path `tst:101-107` (`_write_canonical_diagnostics`) |
| Q1 tasks 8.1–8.7 | confirmed; **8.1–8.3 off on host** | live `configs/selfplay_config.json` = tracked copy (10 keys, no `enable_diagnostics`); `selfplay_orchestrator.py:213` default `False` |
| Q1 stop 2 (no ranked-predictions writer) | confirmed; **host-confirmed** | `predictions/ranked_predictions.json` absent on host (`predictions/` holds `next_draw_prediction.json` and `history/` only). `diagnostics_outputs/soak_report_real.json` (2026-02-15) recorded it as "exists" then — a hand-placed input, consistent with the audit's 8.7 row |
| Q2, Q3, Q4, Q5, Q6 | confirmed | anchors unchanged |
| Q7 rows 1–11 | confirmed, row 3 and the trace refined (C3) | |
| B1 | confirmed; **host-observed**; consequence refined (C4) | A.1 Q1 |
| B2 | confirmed; **host-observed** | 60/60 complete NN reports carry `model_specific.gradient_health.gradient_spread_ratio` = 1.0; none carries `feature_gradient_spread` |
| B3 / B4 | confirmed; host-consistent | 0 xgboost/lightgbm reports on host; the 4 CatBoost archives are dated 2026-02-09, before `27947c6` (S88, 2026-02-14) |
| B5 | confirmed; **host-observed** | CatBoost archive `catboost_20260209_025838.json` top-10 features named `f4`, `f6`, `f9`, `f3` |
| B6 | confirmed; **host-observed** | no `diagnostics_outputs/compare_models_summary.json`; 60 `compare_models_summary_S88_*.json` files, newest `…20260308_030811.json`, keys include `winner_best_effort`, no `winner_model_type`/`winner`. Reader introduced at `ca0b2bf` (S93, 2026-02-16) |
| B7 | confirmed | reader keys `diagnostics_llm_analyzer.py:183-191` |
| B8, B9 | confirmed; host-consistent | A.0 |
| B10–B17 | confirmed | anchors unchanged; B12 `record_event` has existed since `87c5bf1` (S83, 2026-02-13) and `LearningTelemetry` defines no such method (`modules/learning_telemetry.py` `def` list) |
| S1–S10 | **10 confirmed** | anchors unchanged |
| D1–D18 | **18 confirmed**; D5 anchor corrected (C1); D7 extended | D7: the config file is absent on host (A.0) |
| §8 governance counts | corrected (C2) | |

**Corrections (4).**
- **C1** — D5 and Q7: `build_diagnostics_prompt` is at `diagnostics_llm_analyzer.py:112`, not `:111` (`:111` is blank).
- **C2** — §8 says `docs/TB_RULING_*` = 24 files. The live tree has **30** (`ls docs/TB_RULING_* | wc -l`). None of the 30 mentions `training_diagnostics`, `training_health`, Chapter 14, the skip registry or `per_survivor_attribution` (`/bin/grep -l -i`). Two proposals not cited by the audit touch the subject and were read here: `PROPOSAL_CATEGORY_B_NN_TRAINING_ENHANCEMENTS_v1_0.md:27` ("Phase A (Diagnose) is complete"; `:357` asks whether the skip threshold should drop to 1) and `PROPOSAL_EPISTEMIC_AUTONOMY_UNIFIED_v1_3.md:36`, `:66`, `:148-166` (Chapter 14 is advisory, WATCHER is the sole decision authority). Neither changes a label.
- **C3** — Q7 trace, "the stale NN file stays on disk and keeps being graded". **Partly corrected.** WATCHER deletes `diagnostics_outputs/training_diagnostics.json` before each RETRY (`agents/watcher_agent.py:2897-2901`, `[S99]`), and the file is absent on the host now. But the host archive shows stale grading did happen: of 78 archived checks that carry a producer `generated_at`, only **43 distinct reports** were graded — **35 checks re-graded a report that had already been graded**. Example: the check archived `2026-03-07T04:11:26Z` graded a report generated `2026-02-25T21:06:08`; the check archived `2026-03-08T03:07:51Z` graded the same report (`generated_at 2026-03-06T21:32:40`) as the check of `2026-03-07T05:32:40Z`. The deletion covers the RETRY path only; a run that ends in PROCEED, PROCEED_WITH_NOTE or SKIP_MODEL leaves the file for the next run.
- **C5** — Q1 row 8.6: `TREND_WARNING_THRESHOLD = 0.3` is at `test_phase_8_soak.py:53`, not `:54` (`:54` is `HIT_REGRESSION_DELTA`). Also `mpo:171-172` used in A.1 Q2 was corrected to `:174` (rule) / `:178` (filter) before this addendum was closed.
- **C4** — B1 consequence, "Any diagnostics file that exists is graded critical". True of every archived check on the host (107/107). But with the producer's key, 26 of 64 real reports would not be critical (A.1 Q1). The repair statement in B1 stands; the claim that the key alone explains every critical grade does not.

**Resolved [UNVERIFIED] items (3 of 5).**
1. **NN per-feature gradients are never captured in production — RESOLVED, confirmed.** 0 of 75 real schema-1.1.0 NN reports (60 complete + 14 partial + the current `neural_net_diagnostics.json`) contain `feature_gradients` or `top_10`; every complete one carries `gradient_spread_ratio: 1.0`, the initial value at `training_diagnostics.py:563`, which `:595-598` overwrites only when the last round has `feature_gradients`. The mechanism (`grad_input[0]` is `None` for an input that does not require grad, `td:430`) is still an inference; the outcome is observed.
2. **`dead_pct` ≈ 0 on every layer — RESOLVED, confirmed.** 242 of 242 `layer_health` entries in those reports have `dead_pct` 0.0.
3. **Steady-state NN exclusion — RESOLVED.** A.1 Q2.
4. **Tracker soak results — still [UNVERIFIED].** `diagnostics_outputs/soak_report_real.json` (2026-02-15, 5/5 passed) was read; it reports `features=62` for the loaded model and a `ranked_predictions.json` that no longer exists, so it describes inputs production does not produce now. Not replayable read-only.
5. **Tree `Booster` objects lacking `feature_importances_` — still [UNVERIFIED].** Not executed (no runs). Host-consistent: no xgboost/lightgbm report exists.

**Newly resolved UNAVAILABLE surfaces.** Git history (884 commits; entry commits for B1, B2, B6, B12 and the NN-only `_cli_enable_diagnostics` at `dd34310`, 2026-02-15, given above); gitignored files and host state (A.0, A.1); `agent_manifests/definitions.json` (below).

### A.3 New observations (recorded, not labelled as chapter claims)

- **N1 — Governance attribution of the NN criticals.** `SESSION_CHANGELOG_20260225_S111.md:89-94` attributes the NN's repeated criticals to un-normalised `y`; `SESSION_CHANGELOG_20260307_S121.md:45` resets the registry after the fix. The two checks after S121 (`2026-03-07T05:32:40Z`, `2026-03-08T03:07:51Z`) are critical and both carry "Early stop ratio 0.00", as do all checks before. The changelogs never name the key mismatch; S92 (`SESSION_CHANGELOG_20260215_S92.md:188`) names the symptom only. This is the same B1 — recorded here so a reader does not take the S121 reset as evidence that the grade was fixed.
- **N2 — `agent_manifests/definitions.json`** (gitignored, 6,635 B, mtime 2026-01-01 19:26 -0800, schema 1.1.0, `updated_at` 2026-01-02). Read in full. It declares no training-diagnostics output, health check, skip registry or `enable_diagnostics`. Its Step-5 `sidecar_schema.example` still shows `feature_count: 62` and `target_name: "holdout_hits"` (stale against `mpo:1482` and `mpo:612`; same staleness as D1 and Q2), and its Step-6 `outputs.ranked` names `predictions/ranked_survivors.json`, not the `predictions/ranked_predictions.json` Chapter 13 reads (`c13:876`). **No tracked `*.py` reads `definitions.json`** (`/bin/grep -rn 'definitions.json' --include=*.py`), so none of this reaches a code path.
- **N3 — Other diagnostic artifacts on the host, for completeness:** `diagnostics_outputs/llm_proposals/` 44 files (newest `diagnostics_analysis_20260308_030804.json`) — the S1 archive with no reader; `diagnostics_outputs/holdout_feature_autocorr.json` (2026-02-25 16:57) — written once, presumably by hand, since the audit found no pipeline caller (`mpo:59`); `strategy_recommendation.json` at repo root (2026-02-07) — produced by the Strategy Advisor, not by any Chapter 14 bridge; `diagnostics_history/` 233 files — Chapter 13 post-draw diagnostics, the directory `parameter_advisor.py` reads.
- **N4 — Host services:** `crontab -l` → none for michael; no systemd unit for WATCHER or diagnostics (`systemctl list-units`). Nothing on the host runs the health check on a schedule.

### A.4 Tally

```
Existing audit claims re-verified           72  (22 accurate + 17 B + 10 S + 18 D + 5 U)
  confirmed as written                      65
  confirmed with consequence corrected       2  (C3 — Q7 stale-file trace; C4 — B1 consequence)
  [UNVERIFIED] resolved                      3  (#1, #2, #3 — all confirmed the audit's suspicion)
  [UNVERIFIED] remaining                     2  (#4, #5)
Non-claim corrections                        3  (C1, C5 anchors; C2 governance file count)
Labels changed                               0 on existing items
New label                                    1  B18 BROKEN — UNVERIFIED #1, once resolved, is a
                                                code path that does not do what CH14 §3.1 says
                                                (no governance record found; S172 trace report
                                                :426 describes the field as present)
```

**Resulting chapter counts** (written into `CH14` §16): 22 accurate · 18 BROKEN (B1–B18) · 10 SWITCHED-OFF-BY-RULING · 18 DESCRIPTION-ONLY (D12 now host-observed) · 2 UNVERIFIED.

**Count reconciliation (added 2026-11-11, after commit `741822b`).** The line above sums to **70**, not the 72 claims
re-verified. Nothing was lost. The three resolved UNVERIFIED items went to different places, and only one of them
became a label of its own:

| Item | Resolution | Final placement |
|---|---|---|
| U1 NN per-feature gradients never captured | confirmed: 0/75 real reports | **became B18** (BROKEN). This is why BROKEN goes 17 → 18; B18 is not a 73rd claim |
| U2 `dead_pct` ≈ 0 on every layer | confirmed: 242/242 entries | **absorbed into D12**: host observation of an item already counted as DESCRIPTION-ONLY, not a separate claim |
| U3 steady-state NN exclusion | resolved (A.1 Q2): `neural_net` is **not** excluded today (count 1 < 3, after hand resets) | **absorbed into B1**: the host-observed consequence of the key mismatch plus the registry history, not a separate claim |

```
Accurate                  22
BROKEN                    18   (B1–B17 + B18, which was U1)
SWITCHED-OFF-BY-RULING    10
DESCRIPTION-ONLY          18
UNVERIFIED                 2   (U4 soak results, U5 Booster attributes)
Absorbed into an item      2   (U2 → D12; U3 → B1), not counted separately
-------------------------------
Total                     72
```

### A.5 Surfaces read in this addendum

- **Tracked code:** `training_health_check.py`, `training_diagnostics.py`, `train_single_trial.py`, `meta_prediction_optimizer_anti_overfit.py`, `agents/watcher_agent.py`, `per_survivor_attribution.py`, `chapter_13_orchestrator.py`, `selfplay_orchestrator.py`, `inner_episode_trainer.py`, `diagnostics_llm_analyzer.py`, `modules/learning_telemetry.py`.
- **Tracked config/state:** `watcher_policies.json` (`training_diagnostics` block), `agent_manifests/reinforcement.json` (`enable_diagnostics` at `:78`, `:179`), `configs/selfplay_config.json`, `diagnostics_outputs/model_skip_registry.json` and its 9 committed versions.
- **Gitignored / host:** `agent_manifests/definitions.json`; `diagnostics_outputs/` (all files, incl. `neural_net_diagnostics.json`, 111 `history/` files, 60 compare summaries, `llm_proposals/`, `soak_report_real.json`); `diagnostics_history/` (listing); `predictions/` (listing); `strategy_recommendation.json` (keys); `find / -xdev` for `*_diagnostics.json`, `model_skip_*`, `compare_models_summary*`, `reinforcement_engine_config*`, `watcher_failures.jsonl`; `crontab -l`; `systemctl list-units`.
- **Governance:** `docs/TB_RULING_*` (30), `docs/PROPOSAL_*`, `docs/TEAM_ALPHA_*` by `/bin/grep -l -i`; `TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md` (anchors), `SESSION_CHANGELOG_20260215_S90.md`, `…S92.md`, `…S93.md`, `…S94.md`, `…S95.md`, `…S111.md`, `…S121.md`, `S111_TEAM_BETA_BRIEFING.md`, `CHAPTER_13_IMPLEMENTATION_PROGRESS_v3_9.md`, `PROPOSAL_CATEGORY_B_NN_TRAINING_ENHANCEMENTS_v1_0.md`, `PROPOSAL_EPISTEMIC_AUTONOMY_UNIFIED_v1_3.md`, `TODO_CONSOLIDATED.md` (Ch14 rows).
- **UNAVAILABLE:** the private-repo history beyond this clone's 884 commits (none needed); ser8 archives; any rig; execution of xgboost/lightgbm Booster attribute behaviour (not run, by scope); WATCHER run logs for the 2026-03-07/08 checks (no log file searched for or found under `diagnostics_outputs/`; `[UNVERIFIED]` which WATCHER action followed each).
