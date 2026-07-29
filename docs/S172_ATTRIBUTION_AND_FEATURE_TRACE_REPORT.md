# Source Trace Report — Survivor Feature Schema + Per-Survivor Attribution

**Box:** VM101 `/home/michael/distributed_prng_analysis`
**Date:** 2026-07-28
**Mode:** read-only investigation — no files modified, no pipeline launched, no commits
**Scope:** two independent questions, answered from live source on VM101, every claim anchored to `file:line`

---

# QUESTION 1 — Does the survivor feature vector derive entirely from the 22-array NPZ contract?

## Headline answer

**No.** The final vector is **91 features** (not ~62). Only **13 of the 22 NPZ arrays** are consumed as features. **19 of the 91 features are category (c)** — they do not come from the 22 arrays at all:

- **14** `global_*` features from `GlobalStateTracker(train_history)` — computed from the **draw history**, survivor-independent
- **5** permanently-zero placeholders (`skip_mean`, `skip_std`, `skip_entropy`, `survivor_velocity`, `velocity_acceleration`) — **no producer exists anywhere in the repo**

**RANGE-MINER implication: the miner must emit exactly the 22 arrays and nothing more.** Every category-(c) feature is either survivor-independent (injected identically for all survivors from `train_history`) or dead (hardcoded 0.0 with no upstream writer). No miner-side change is needed to satisfy the current Step 3 contract.

## 1.1 The frozen 22-array contract, verbatim

`convert_survivors_to_binary.py:50-73` (`_EMPTY_NPZ_DTYPES`), in `savez` order:

| # | Array | dtype |
|---|---|---|
| 1 | `seeds` | `np.uint32` |
| 2 | `forward_matches` | `np.float32` |
| 3 | `reverse_matches` | `np.float32` |
| 4 | `window_size` | `np.int32` |
| 5 | `offset` | `np.int32` |
| 6 | `trial_number` | `np.int32` |
| 7 | `skip_min` | `np.int32` |
| 8 | `skip_max` | `np.int32` |
| 9 | `skip_range` | `np.int32` |
| 10 | `forward_count` | `np.float32` |
| 11 | `reverse_count` | `np.float32` |
| 12 | `bidirectional_count` | `np.float32` |
| 13 | `intersection_count` | `np.float32` |
| 14 | `intersection_ratio` | `np.float32` |
| 15 | `intersection_weight` | `np.float32` |
| 16 | `bidirectional_selectivity` | `np.float32` |
| 17 | `forward_only_count` | `np.float32` |
| 18 | `reverse_only_count` | `np.float32` |
| 19 | `survivor_overlap_ratio` | `np.float32` |
| 20 | `score` | `np.float32` |
| 21 | `skip_mode` | `np.uint8` |
| 22 | `prng_type` | `np.uint8` |

The non-empty write path emits the same 22 names in the same order — `convert_survivors_to_binary.py:201-225`. The six `*_count` arrays are `float32` despite being logically integral; the header notes this is deliberate contract reproduction (`convert_survivors_to_binary.py:48-49`).

Supporting context on the contract:

- The empty-artifact path writes all 22 arrays at length 0 so an empty NPZ is structurally identical to a non-empty one except for length — `convert_survivors_to_binary.py:94-110`. The prior one-array (`seeds=[]`) form is called out as a defect, not an alternate representation (`:97-101`).
- The metadata sidecar independently records `"array_count": 22` — `convert_survivors_to_binary.py:249`, with the arrays grouped as `core` (3), `metadata_int` (6), `metadata_float` (11), `categorical` (2) at `:238-248`.
- `skip_mode` / `prng_type` are encoded through the canonical registry seam `utils/prng_encoding` (`convert_survivors_to_binary.py:38-43`, applied at `:178-186`), which hard-fails on an unknown identity rather than silently mapping to 0 (`:30-37`).
- `skip_range` is normalized from int, `[min, max]` list, or `"min-max"` string by `_parse_skip_range()` — `convert_survivors_to_binary.py:139-155`.

## 1.2 Step 3 trace: NPZ → chunk → feature vector

1. **Load** — `generate_step3_scoring_jobs.py:181` calls `load_survivors()` (`utils/survivor_loader.py:103-109`), which returns NPZ arrays natively (detection order documented at `utils/survivor_loader.py:113-118`; `return_format="auto"` → NPZ→array, `:124-126`).
2. **Rectangularize** — `extract_survivors_full()` at `generate_step3_scoring_jobs.py:62-102` transposes arrays to per-survivor dicts. It iterates **all** NPZ keys (`:79`, `:83`), keeps any array whose length matches `n` (`:85`), converts numpy scalars to Python natives (`:88-89`), and renames `seeds`→`seed` (`:90-91`). **All 22 arrays survive into the chunk file** — nothing is filtered here. The docstring records the provenance of this fix: *"Previous version discarded metadata, causing 14/47 ML features = 0"* (`:66-68`).
3. **METADATA LOSS guardrail** — `generate_step3_scoring_jobs.py:95-100`: raises `ValueError` if `len(result[0]) < 3`. Note this is a **weak** guard: the message claims `"Expected 20+"` (`:99`) but the threshold is `3` (`:96`), so dropping 19 of 22 fields would pass silently. It only catches total collapse, not the partial loss it was written for.
4. **JSON / list fallback path** — `generate_step3_scoring_jobs.py:104-117` handles a list-of-dicts input, aliasing `candidate_seed`→`seed` (`:109-110`), and degrades plain integers to `{'seed': int(s)}` minimal dicts (`:113-115`). The guardrail at `:95-100` does **not** cover this branch.
5. **Chunk write** — `generate_step3_scoring_jobs.py:211-229`: `chunk_list()` splits full survivor objects (`:212`, helper at `:48-50`), chunk dir `scoring_chunks` created at `:217-218`, chunks written as `chunk_{i:04d}.json` at `:228-229`. Field count is echoed for visibility at `:189-191`.
6. **Worker load** — `full_scoring_worker.py:581-584`: `survivor_metadata = {s['seed']: s for s in survivors_full}` — seed-keyed metadata map. Comment at `:592-595` notes the metadata is already in the chunk file, so no large-file load is needed.
7. **Extraction** — `full_scoring_worker.py:375-381` calls `scorer.extract_ml_features_batch(seeds=..., lottery_history=train_history, forward_survivors=..., reverse_survivors=..., survivor_metadata=survivor_metadata)`. Passed through from `main()` at `:651`.
8. **Metadata merge** — `survivor_scorer.py:770-781`: for each seed, **18 named fields** are pulled from metadata if present and non-`None`.
9. **Global injection** — `full_scoring_worker.py:403-405`: 14 `GlobalStateTracker` features merged with a `global_` prefix.

Scorer construction and configuration:

- `scorer = scorer_class(prng_type=prng_type, mod=mod)` — `full_scoring_worker.py:344`
- `SurvivorScorer.__init__` signature — `survivor_scorer.py:90`
- `self.residue_mods` resolution — `survivor_scorer.py:107`, default `DEFAULT_RESIDUE_MODS = [8, 125, 1000]` at `survivor_scorer.py:74`
- `self.temporal_window_size` / `self.temporal_num_windows` — `survivor_scorer.py:109-110`, defaults `DEFAULT_TEMPORAL_WINDOW = 100` / `DEFAULT_TEMPORAL_WINDOWS = 5` at `survivor_scorer.py:76-77`

## 1.3 Enumeration of all 91 features by origin

**Empirically confirmed against live output**: `full_scoring_results/full_scoring_results_20260311_170236/chunk_0000.json` → `len(features) == 91`, 84 records. Per-record top-level keys: `['seed', 'score', 'features', 'metadata', 'holdout_hits', 'holdout_features', 'holdout_quality']`.

### (a) Directly from an NPZ array — **13 features**

Merged at `survivor_scorer.py:770-781` (batch path). The merge list is enumerated at `survivor_scorer.py:774-779`:

`forward_count`, `reverse_count`, `bidirectional_count`, `intersection_count`, `intersection_ratio`, `intersection_weight`, `bidirectional_selectivity`, `forward_only_count`, `reverse_only_count`, `survivor_overlap_ratio`, `skip_min`, `skip_max`, `skip_range`

Live variance check on `chunk_0000.json` (84 records) — all 13 carry real, non-zero, varying values:

| Feature | unique values | sample |
|---|---|---|
| `forward_count` | 2 | 14296.0, 15711.0 |
| `reverse_count` | 2 | 3649.0, 14565.0 |
| `bidirectional_count` | 2 | 21.0, 63.0 |
| `intersection_count` | 2 | 21.0, 63.0 |
| `intersection_ratio` | 2 | 0.0010858885943889618, 0.002187651814892888 |
| `intersection_weight` | 2 | 0.0010847107041627169, 0.002182876458391547 |
| `bidirectional_selectivity` | 2 | 0.9815310835838318, 4.305562973022461 |
| `forward_only_count` | 2 | 14233.0, 15690.0 |
| `reverse_only_count` | 2 | 3628.0, 14502.0 |
| `survivor_overlap_ratio` | 2 | 0.0013366431230679154, 0.004406827036291361 |
| `skip_min` | 1 | 5.0 |
| `skip_max` | 1 | 56.0 |
| `skip_range` | 1 | 51.0 |

(Two distinct values reflect two source trials in that chunk; `skip_*` are trial-constant by construction. All are genuinely populated, not defaulted.)

### (b) Deterministically computed in Step 3 — **59 features**

All are derived from the **regenerated PRNG sequence** (seeded by NPZ `seeds`) compared against `train_history`. The only NPZ input is `seeds`; the comparison target is the draw history, which is **not** survivor data.

PRNG regeneration, batch path — `survivor_scorer.py:566-582`:
- `seeds_t = torch.tensor(seeds, ...)` — `:566`
- GPU reference kernel dispatch — `:570-575`
- CPU fallback `self._cpu_batch_generate(seeds, n)` — `:578`, `:581`
- broadcast of history to `(batch_size, n)` — `:586`

PRNG regeneration, sequential path — `survivor_scorer.py:355` (`seq = self._generate_sequence(seed, n, skip=skip)`), tensors at `:361-363`.

**Match / statistics block — 27** (`survivor_scorer.py:702-731`):

| Feature | Line | Derivation |
|---|---|---|
| `score` | `:703` | `base_scores * 100`, where `base_scores = match_counts / n` (`:592-593`) |
| `confidence` | `:704` | `clamp(base_scores, min=self.min_confidence_threshold)` |
| `exact_matches` | `:705` | `matches.sum(dim=1)` (`:592`) |
| `total_predictions` | `:706` | `float(n)` |
| `best_offset` | `:707` | constant `0.0` |
| `pred_mean` | `:708` | `predictions.float().mean(dim=1)` (`:596`) |
| `pred_std` | `:709` | `:597` |
| `pred_min` | `:710` | `:600` |
| `pred_max` | `:711` | `:601` |
| `residual_mean` | `:712` | residuals `= predictions - hist` (`:602`), mean `:603` |
| `residual_std` | `:713` | `:604` |
| `residual_abs_mean` | `:714` | `:605` |
| `residual_max_abs` | `:715` | `:606` |
| `actual_mean` | `:716` | `hist_t.float().mean()` (`:608`) |
| `actual_std` | `:717` | `:609` |
| `lane_agreement_8` | `:718` | `:612` |
| `lane_agreement_125` | `:719` | `:613` |
| `lane_consistency` | `:720` | `(lane_8 + lane_125) / 2` (`:614`) |
| `hundreds_digit_agreement` | `:722` | `:618` |
| `tens_digit_agreement` | `:723` | `:619` |
| `ones_digit_agreement` | `:724` | `:620` |
| `expected_digit_match_count` | `:725` | `_hd + _td + _od`, range 0.0–3.0 (`:621`) |
| `temporal_stability_mean` | `:726` | window scores `:672-678`, mean `:683` |
| `temporal_stability_std` | `:727` | `:684` |
| `temporal_stability_min` | `:728` | `:685` |
| `temporal_stability_max` | `:729` | `:686` |
| `temporal_stability_trend` | `:730` | per-seed regression slope `:688-690` |

Digit features are documented as S119 / CA Lottery spec 03:00-09r, additive alongside CRT lanes — `survivor_scorer.py:616-617` (batch) and `:426-428` (sequential).

**Residue block — 9** (`survivor_scorer.py:624-667`, `residue_mods = [8, 125, 1000]` at `:74`, loop at `:625`):

`residue_8_match_rate`, `residue_8_coherence`, `residue_8_kl_divergence`, `residue_125_match_rate`, `residue_125_coherence`, `residue_125_kl_divergence`, `residue_1000_match_rate`, `residue_1000_coherence`, `residue_1000_kl_divergence`

- match rate — `:630-631`
- vectorized batch histograms via `scatter_add_` — `:636-656`
- normalized distributions — `:659-660`
- KL divergence — `:663`; coherence `1/(1+kl)` — `:664`; stored `:666-667`
- merged into the result tensor dict at `:734`

Sequential equivalent: `survivor_scorer.py:377-390` (uses `scipy.stats.entropy` at `:387` rather than the inline PyTorch form).

**Battery Tier 1A — 23** (`survivor_scorer.py:745-757`; producer `compute_battery_features()` at `:188-200`; sequential call at `:451-452`):

`batt_fft_peak_mag`, `batt_fft_secondary_peak`, `batt_fft_spectral_conc`, `batt_fft_diff_peak`, `batt_fft_diff_conc`, `batt_ac_lag_01`, `batt_ac_lag_02`, `batt_ac_lag_03`, `batt_ac_lag_04`, `batt_ac_lag_05`, `batt_ac_lag_06`, `batt_ac_lag_07`, `batt_ac_lag_08`, `batt_ac_lag_09`, `batt_ac_lag_10`, `batt_ac_decay_rate`, `batt_ac_sig_lag_count`, `batt_cs_max_excursion`, `batt_cs_mean_excursion`, `batt_cs_zero_crossings`, `batt_bf_hamming_mean`, `batt_bf_hamming_std`, `batt_bf_popcount_bias`

- Introduced as S113, "23 columns" — `survivor_scorer.py:449`, `:754`
- **Leakage invariant** — `survivor_scorer.py:192-193`: *"CRITICAL INVARIANT: seq is the PRNG output array only. Never pass lottery_history here — that is a leakage violation."* Enforced in practice: batch path passes `predictions_np[i]` (`:756`), sequential passes `seq` (`:451`), with the comment *"Seq invariant: seq was generated from seed, NOT from lottery_history"* (`:450`).
- Zero-fill fallback if the GPU→CPU transfer of predictions fails — `:758-768`
- Canonical name list also appears in `_empty_ml_features()` — `survivor_scorer.py:481-489`

**Important:** NPZ `score` (array 20) is **not** used. The feature named `score` is recomputed from the regenerated sequence at `survivor_scorer.py:703`. The converter's own comment (`convert_survivors_to_binary.py:162`) describes the NPZ `score` as `avg(fwd_rate, rev_rate)` in v3.0+ — a different quantity from the recomputed match-rate score.

### (c) From a source outside the 22 arrays — **19 features**

**c.1 — Global state features (14).**

- Import — `full_scoring_worker.py:82` (`from models.global_state_tracker import GlobalStateTracker, GLOBAL_FEATURE_NAMES`), described at `:81` as "GPU-neutral module for global features (14 features)"
- Construction from draw history — `full_scoring_worker.py:348`: `global_tracker = GlobalStateTracker(train_history, {"mod": mod})`
- Single evaluation — `full_scoring_worker.py:349`: `global_features = global_tracker.get_global_state()`; count logged `:350`
- Merge with prefix, batch path — `full_scoring_worker.py:403-405`
- Merge with prefix, fallback path — `full_scoring_worker.py:458-460`

Names — `models/global_state_tracker.py:366-381`:

`frequency_bias_ratio`, `high_variance_count`, `marker_390_variance`, `marker_575_variance`, `marker_804_variance`, `power_of_two_bias`, `regime_age`, `regime_change_detected`, `reseed_probability`, `residue_1000_entropy`, `residue_125_entropy`, `residue_8_entropy`, `suspicious_gap_percentage`, `temporal_stability`

→ emitted as `global_frequency_bias_ratio` … `global_temporal_stability` (14 keys).

Class definition at `models/global_state_tracker.py:84`. Self-test asserts name/count/dtype stability at `:401-409`.

Source: **draw history only.** Computed once per worker process (`full_scoring_worker.py:348-349`, outside the per-seed loop) and stamped identically onto every survivor (`:404-405`) — constant across the population by construction, carrying zero per-survivor signal.

**c.2 — Dead placeholders (5).**

Requested from metadata at `survivor_scorer.py:776-778` but **not present in the NPZ contract**, so they fall through to the `setdefault(k, 0.0)` block at `survivor_scorer.py:784-791`:

`skip_mean`, `skip_std`, `skip_entropy`, `survivor_velocity`, `velocity_acceleration`

Sequential path has the same zero-fill list — `survivor_scorer.py:455-461`.

Confirmed dead in live data — `chunk_0000.json`, all 84 records:

| Feature | unique values | value |
|---|---|---|
| `skip_mean` | 1 | 0.0 |
| `skip_std` | 1 | 0.0 |
| `skip_entropy` | 1 | 0.0 |
| `survivor_velocity` | 1 | 0.0 |
| `velocity_acceleration` | 1 | 0.0 |

Confirmed **no producer exists.** A repo-wide search for these names (excluding `backups/`, `step6_restoration/`) returns only consumers, declaration lists, and test/sample fixtures:

- consumers / declaration lists: `feature_importance.py:104-105`, `feature_drift_tracker.py:206-207`, `feature_importance_interpreter.py:115`, `survivor_scorer.py:455-456`, `survivor_scorer.py:776-778`, `survivor_scorer.py:784-785`
- hardcoded sample values (not production writes): `training_diagnostics.py:1035`, `diagnostics_llm_analyzer.py:568`, `feature_drift_tracker.py:620`, `feature_drift_tracker.py:634`
- tests: `test_feature_importance.py:191`, `:195`, `:309`, `:360-361`; `run_tests.py:129`
- patch scripts (historical): `apply_s113_battery_tier1a.py:231`
- unrelated: `trse_step0.py:16`, `:61`, `:483`, `:683`, `:687`, `:728` — `analyze_skip_entropy()` is a draw-level TRSE analysis emitting a `skip_entropy_profile` dict, consumed at `window_optimizer_bayesian.py:524`; it is **not** a per-survivor feature producer and does not write `skip_entropy` into a survivor record.

**Nothing writes any of these five into a survivor record anywhere in the tree.**

**Count check:** 13 (a) + 59 (b) + 19 (c) = **91** ✓

Cross-check of the arithmetic against the source structure: 27 (match/stats, `:702-731`) + 9 (residue, `:624-667`) + 23 (battery, `:745-757`) = 59 computed; + 13 NPZ-merged + 5 dead placeholders = 77 scorer-level features; + 14 `global_*` = **91**. The scorer's own log line reports the pre-battery, pre-metadata count via `len(self._empty_ml_features())` at `survivor_scorer.py:563`.

## 1.4 NPZ arrays that never reach the feature vector

Nine of 22 are carried into the chunk file (`generate_step3_scoring_jobs.py:83-92` preserves everything) but never merged as features (absent from the merge list at `survivor_scorer.py:774-779`):

| Array | Fate |
|---|---|
| `seeds` | join key + PRNG regeneration input (`survivor_scorer.py:566`, `:355`) — drives all 59 category-(b) features, but is not itself a feature |
| `forward_matches` | **unused** — not in the merge list at `survivor_scorer.py:774-779` |
| `reverse_matches` | **unused** |
| `window_size` | **unused** |
| `offset` | **unused** |
| `trial_number` | **unused** |
| `score` | **unused as a feature** — the `score` feature is recomputed at `survivor_scorer.py:703` |
| `skip_mode` | **unused** |
| `prng_type` | **unused as a feature** — passed separately as a scorer constructor argument (`full_scoring_worker.py:344`) and recorded in per-record `metadata` (`:396`) |

Worth flagging: `forward_matches` / `reverse_matches` are described in the converter header (`convert_survivors_to_binary.py:16-20`) as *"the surface fingerprint signals that ML uses to rank survivors"* — the v3.1.0 fix that remapped them from trial-level aggregates (`forward_count`/`reverse_count`) to per-seed match rates (`forward_match_rate`/`reverse_match_rate`), with the stated rationale that the old mapping made *"all quality fields identical for every seed in the same trial"* (`:18-19`). The write path is `convert_survivors_to_binary.py:123-131`, and Step 1 integration v3.0+ is required or a warning is emitted (`:114-117`).

**Those two arrays do not appear in the feature vector.** Either the header comment is stale or the merge list at `survivor_scorer.py:774-779` is missing them. This is a discrepancy in the current code, not something RANGE-MINER causes or fixes; the miner should keep emitting both.

## 1.5 The "~62" in the docs, and the real number

**Real number: 91**, confirmed empirically from `full_scoring_results/full_scoring_results_20260311_170236/chunk_0000.json`.

The **~62** almost certainly traces to `feature_importance.py`: `STATISTICAL_FEATURES` (46 entries, `feature_importance.py:95-111`) + `GLOBAL_STATE_FEATURES` (14 entries, `:113-119`) = **60**.

That canonical list is **stale by 31 features.** Diffing the list (with `global_` prefixes applied) against the live 91: everything in the canonical list is present live (`canonical − live = {}`), and these 31 live features are **missing from the list**:

- **23 battery** (S113): `batt_ac_decay_rate`, `batt_ac_lag_01` … `batt_ac_lag_10`, `batt_ac_sig_lag_count`, `batt_bf_hamming_mean`, `batt_bf_hamming_std`, `batt_bf_popcount_bias`, `batt_cs_max_excursion`, `batt_cs_mean_excursion`, `batt_cs_zero_crossings`, `batt_fft_diff_conc`, `batt_fft_diff_peak`, `batt_fft_peak_mag`, `batt_fft_secondary_peak`, `batt_fft_spectral_conc`
- **4 digit-agreement** (S119): `hundreds_digit_agreement`, `tens_digit_agreement`, `ones_digit_agreement`, `expected_digit_match_count`
- **4 metadata**: `bidirectional_count`, `bidirectional_selectivity`, `skip_min`, `skip_max`

The same stale 46-name list is duplicated at `feature_drift_tracker.py:206-207` (partial region observed), so any drift/importance reporting keyed off these constants under-covers the live vector by 31 columns.

**Independent corroboration from the trained model sidecar** — `models/reinforcement/best_model.meta.json`:

- top-level keys: `['schema_version', 'model_type', 'checkpoint_path', 'checkpoint_format', 'feature_schema', 'signal_quality', 'data_context', 'training_metrics', 'hyperparameters', 'optuna', 'hardware', 'training_info', 'agent_metadata', 'provenance']`
- `model_type`: `"neural_net"`
- `feature_schema.feature_count`: **89**; `len(feature_schema.feature_names)`: **89**
- `feature_schema.ordering`: `"lexicographic_by_key"`
- `feature_schema.source_file`: `/home/michael/distributed_prng_analysis/survivors_with_scores.json`
- `feature_schema.excluded_features`: `['score', 'confidence', 'holdout_hits', 'holdout_quality']`

Diffing the sidecar's 89 names against the live 91: `live − model = {score, confidence}`, `model − live = {}`. So **91 extracted − 2 leakage-excluded = 89 trained**, exactly consistent. (`holdout_hits` and `holdout_quality` are listed as excluded but were never inside `features` to begin with — they are sibling top-level keys.)

**Other per-record fields outside `features`** (not part of the 91):

- `holdout_hits` — `full_scoring_worker.py:407` (batch), `:467` (fallback); computed by `compute_holdout_hits_batch()` (`:274`, invoked `:364-370`), described as the y-label for Step 5 (`:406`), with `train_history_len` marked `DERIVED - not configurable` (`:367`)
- `holdout_quality` — `full_scoring_worker.py:423` (batch), `:484` (fallback); from `compute_holdout_quality()` imported at `:65`
- `holdout_features` — `full_scoring_worker.py:419-422` (batch), `:480-483` (fallback): the same scorer run against `holdout_history` instead of `train_history` via `_s111_extract_features_with_optional_skip()` (`:68-79`), which uses `inspect.signature()` to pass `skip` only if supported (`:74-76`) — harness-discipline compliant. **Live count: 77** = 91 − 14, since `global_*` are not re-merged onto the holdout dict. Zero-filled on exception or when `holdout_history` is empty (`:426-430`, `:487-491`)
- `metadata` — `full_scoring_worker.py:395-401`: `prng_type`, `mod`, `worker_hostname`, `worker_gpu`, `timestamp`
- `error` — `full_scoring_worker.py:495` on double failure, with `features: {}`

The worker reports its own count from the first valid result — `full_scoring_worker.py:698-699` (`feature_count = len(valid_results[0].get('features', {}))`), surfaced in the output at `:710` and logged at `:726`. Note the module docstring and argparse help still say **50** features (`full_scoring_worker.py:5`, `:8`, `:12`, `:24-27`, `:325`, `:338`, `:508`) — also stale relative to the live 91.

## 1.6 One inconsistency worth recording

The **batch path** merges 18 metadata fields (`survivor_scorer.py:774-779`); the **sequential fallback path** merges only 6 (`full_scoring_worker.py:453-454`: `forward_count`, `reverse_count`, `bidirectional_count`, `skip_min`, `skip_max`, `skip_range`).

If the GPU batch throws (`full_scoring_worker.py:438-442` catches and falls through to the sequential loop at `:443`), seven real NPZ-backed features silently become 0.0 via the scorer's own zero-fill (`survivor_scorer.py:455-461`):

`intersection_count`, `intersection_ratio`, `intersection_weight`, `bidirectional_selectivity`, `forward_only_count`, `reverse_only_count`, `survivor_overlap_ratio`

This is the same failure class as the Jan 23 2026 incident that the METADATA LOSS guardrail was written for (`generate_step3_scoring_jobs.py:66-68`), on a path the guardrail does not cover — the guardrail runs at job-generation time on chunk contents (`:95-100`), not at feature-merge time in the worker. Failure is logged (`full_scoring_worker.py:439-441`) but produces no schema-level alarm, and the resulting records are structurally valid with 91 keys.

## 1.7 Answer to the RANGE-MINER question

**Are there category-(c) features? Yes — 19 of 91.** Sources:

1. **14 `global_*`** — `GlobalStateTracker(train_history)`, `full_scoring_worker.py:348-349`, merged `:403-405`; names `models/global_state_tracker.py:366-381`. Source is the **draw history**, available to Step 3 independently of the miner, and identical for every survivor in a run.
2. **5 dead placeholders** — `skip_mean`, `skip_std`, `skip_entropy`, `survivor_velocity`, `velocity_acceleration`; requested `survivor_scorer.py:776-778`, zero-filled `:784-791`; **no producer anywhere in the repo**.

**Neither category obliges RANGE-MINER to emit anything beyond the 22 arrays.** The `global_*` block is survivor-independent and draw-history-derived. The 5 placeholders are structurally dead — nothing upstream of Step 3 has ever written them, so a miner that omits them changes nothing observable.

**The 22-array contract is therefore sufficient and complete for RANGE-MINER.** Of those 22: `seeds` is the identity/regeneration key, 13 are merged directly as features, and 8 are carried but unconsumed (with `forward_matches` / `reverse_matches` being unconsumed *despite* being documented as the primary ML ranking signal — flagged in §1.4 as a pre-existing discrepancy the miner must not "fix" by dropping them).

---

# QUESTION 2 — Is per-survivor attribution actually wired, or only designed?

## Headline answer

**Team Beta is confirmed on `training_diagnostics.py`** — that path is batch-averaged, not per-survivor. **But it is not the only path**: `per_survivor_attribution.py` contains a genuine, correctly-implemented single-survivor attribution module, and Chapter 13 does call it with seed identity attached.

**However, that call is a permanent no-op in production**, blocked by two independent preconditions. Every archived artifact on disk is from synthetic test data. Net status: **implemented, invoked, and unreachable.**

## 2.1 Team Beta's finding — CONFIRMED

`training_diagnostics.py:474`:

```python
feat_grads = self._input_gradients.abs().mean(dim=0)
```

Provenance of the tensor:

- `self._input_gradients` initialized `None` — `training_diagnostics.py:341`
- captured in a **backward hook** on the first layer only — `:429-431`: `if layer_name == self._layer_names[0] and grad_input[0] is not None: self._input_gradients = grad_input[0].detach()`
- hook factory `:418-431`, per-layer gradient stats at `:419-427`

`grad_input[0]` has shape `(batch, n_features)`. **`mean(dim=0)` reduces over the batch dimension** → one vector for the whole minibatch, not one per sample.

Consumption of the reduced vector — `training_diagnostics.py:475-487`:
- `top_k = min(10, len(feat_grads))` — `:475`
- `top_indices = feat_grads.argsort(descending=True)[:top_k]` — `:476`
- written as `snapshot["feature_gradients"] = {"top_10": [...], "spread_ratio": ...}` — `:478-487`
- `spread_ratio = feat_grads.max() / (feat_grads.min() + 1e-10)` — `:486`
- failure is swallowed at debug level — `:488-489`

Per-pass state is cleared immediately after the snapshot — `training_diagnostics.py:496-498` (`self._activations.clear()`, `self._gradients.clear()`, `self._input_gradients = None`), so **nothing per-sample is retained**.

**No survivor identity is available on this path at all** — the hook sees tensors, never seeds. The module self-describes as `PASSIVE OBSERVER — Never modifies gradients, weights, or training behavior` (`training_diagnostics.py:11`).

**Team Beta's characterization is exact.**

The same reduction appears in the Chapter 11 global-importance path, `feature_importance.py:489`:

```python
gradients = X_tensor.grad.abs().mean(dim=0).cpu().numpy()
```

after `output.sum().backward()` (`:486`), inside `_extract_gradient_importance()` (`:465-499`). This is aggregate **by design** and correctly labelled — the docstring says *"Gradient saliency importance for PyTorch neural networks"* (`:473`) and notes it does not require `y` (`:475`). Normalization at `:494-497`.

## 2.2 The genuine per-survivor path — refutes "only aggregate exists"

`per_survivor_attribution.py` — *"Per-Survivor Attribution Module — Chapter 14, Phase 2"* (`:3`), version `1.0.1` at `:42` (`"TB review fixes: device auto-detect, zero_grad, sampling, failure tracking"`). Its stated purpose distinguishes it from Chapter 11 explicitly (`:10-12`): global importance answers *"which features matter on average?"*, per-survivor answers *"for THIS specific seed, which features drove its prediction?"*

Declared invariants — `per_survivor_attribution.py:24-28`: all backends return `feature_name → normalized attribution`; scores are absolute values normalized to sum 1.0; best-effort non-fatal (returns `{}` on failure); no side effects on model state.

**No reduction over samples occurs in any backend** — each is called with exactly one survivor's vector:

| Backend | Function | Mechanism | No-reduction evidence |
|---|---|---|---|
| Neural net | `per_survivor_attribution.py:49-127` | input gradients, `grad` or `grad_x_input` | `x = x.unsqueeze(0).requires_grad_(True)` → batch of 1, shape `[1, N_features]` (`:98`); forward `:101`; `prediction.backward(retain_graph=False)` `:104`; `grads = (x.grad * x).squeeze().abs()` (`:108`) or `x.grad.squeeze().abs()` (`:110`); normalize `:113-115`; map to names `:117` |
| XGBoost | `:134-177` | native `pred_contribs` | `features.reshape(1, -1)` (`:160`); `booster.predict(dmatrix, pred_contribs=True)` (`:163`); shape `[1, N+1]` per comment `:164`; `raw = contributions[0][:-1]` drops bias (`:166`); normalize `:167-171` |
| LightGBM | `:184-224` | native `pred_contrib` | `model.predict(features.reshape(1, -1), pred_contrib=True)` (`:204-207`); `raw = contributions[0][:-1]` (`:213`); normalize `:214-218` |
| CatBoost | `:231-273` | native C++ `ShapValues` | `Pool(features.reshape(1, -1), feature_names=...)` (`:255-258`); `model.get_feature_importance(pool, type="ShapValues")` (`:259`); `raw = shap_values[0][:-1]` (`:262`); normalize `:263-267` |

Dispatch table `_ATTRIBUTION_BACKENDS` — `per_survivor_attribution.py:281-286`; unified entry `per_survivor_attribution()` — `:289-316`, unknown `model_type` → warn + `{}` (`:311-314`).

Method rationale — `:63-69`: `grad_x_input` is the default because it handles differently-scaled features more stably; *"No extra graph cost — same backward pass, one extra multiply."*

**Two prior-TB fixes are present and verifiable:**
- **TB Finding #1** — device auto-detected from model parameters rather than hardcoded, to prevent CUDA init in the parent process (GPU isolation invariant, S72): `per_survivor_attribution.py:83-88`, applied at `:97`
- **TB Finding #2** — `model.zero_grad(set_to_none=True)` before backward to prevent accumulation when called in loops: `:93-95`

Model state is restored after the call — `:90-91` (`was_training = model.training; model.eval()`) and `:120-121` (`if was_training: model.train()`), honoring the no-side-effects invariant at `:28`.

**Permutation importance is not offered per-survivor** — correctly so, since it is inherently multi-sample. It exists only in the aggregate module: `_extract_permutation_importance()` at `feature_importance.py:362-371`, PyTorch GPU variant `_extract_permutation_importance_pytorch()` at `:405`, dispatched at `:194-197` and requiring `y` (`:195-196`).

**Model families supported, by module:**

| Module | NN gradients | Tree SHAP / `pred_contribs` | Permutation | Per-survivor? |
|---|---|---|---|---|
| `per_survivor_attribution.py` | ✅ `:49` (`grad`, `grad_x_input`) | ✅ XGB `:134`, LGB `:184`, CatBoost SHAP `:231` | ❌ not offered | ✅ **yes** |
| `feature_importance.py` (Ch 11) | ✅ `:199-202`, `:465-499` | ✅ native `:191-192` | ✅ `:194-197`, `:362`, `:405` | ❌ aggregate |
| `training_diagnostics.py` | ✅ `:474` (batch-mean) | ❌ | ❌ | ❌ aggregate |

Model-type detection is encapsulated in `_detect_model_type()` — `feature_importance.py:220`, called `:180`, with the comment that it lives *"nowhere else in codebase"* (`:176-178`); method resolution `_resolve_method()` handles `auto` at `:272-291`.

## 2.3 Survivor identity / join key

**Exists on the Chapter 13 path only.**

`chapter_13_orchestrator.py:709-715` (missed survivors) and `:732-738` (hit survivors) attach identity alongside each survivor's attribution:

```python
missed_attributions.append({
    'seed': survivor.get('seed'),      # :710
    'rank': survivor.get('rank'),      # :711
    'top_3_features': sorted(attr.items(), key=lambda x: x[1], reverse=True)[:3],   # :712-714
})
```

**`seed` is the join key**, sourced from the prediction records loaded off disk. The expected input schema is documented at `chapter_13_orchestrator.py:670-671`: *"predictions: Ranked predictions list, each with 'hit', 'features', 'seed', 'rank' keys."*

**Lost on the tier path.** `compare_pool_tiers()` reads only `s["features"]` (`per_survivor_attribution.py:387`) and `s.get("prediction", 0)` for ranking (`:367`). It never touches a seed. The reduction is at `:399-402`:

```python
return {
    f: float(np.mean([a.get(f, 0.0) for a in attrs]))
    for f in feature_names
}
```

Output keys are `top_20` / `top_100` / `top_300` / `divergence` / `metadata` (`per_survivor_attribution.py:405-438`, tier defaults `:361-362`, divergence `:416-428`) — **no seed anywhere**. This is a tier aggregate built *out of* per-survivor attributions: genuinely per-survivor at the leaves (`:388-390`), averaged before it reaches disk.

Two TB-driven guards on this function: **Finding #3** sample capping via `max_samples_per_tier` (`:329`, `:345-347`, `:379-382`) and **Finding #5** failure tracking (`:371-373`, `:386`, `:391-394`), surfaced as `attribution_attempts` / `attribution_failures` in metadata (`:433-434`). Short-tier warning at `:408-412`; all-failed fallback returns zeros at `:396-397`.

**Absent entirely** on `training_diagnostics.py` (hook sees tensors only, `:429-431`) and `feature_importance.py` (operates on an `X` matrix with no identity column, `:126-134`).

## 2.4 Output artifacts

| Artifact | Written by | Schema | On disk? |
|---|---|---|---|
| `diagnostics_outputs/history/root_cause_<ts>.json` | `_archive_post_draw_analysis()` — `chapter_13_orchestrator.py:860-873`; dir created `:863-864`, timestamped name `:865-868`, dumped `:869-870` | `type`, `draw_id`, `timestamp`, `missed_count`, `hit_count`, `attribution_success`, `feature_divergence_ratio`, `missed_relied_on`, `hits_relied_on`, `feature_overlap`, `diagnosis`, `missed_details[≤5]`, `hit_details[≤5]` — built at `:766-783`; **`missed_details`/`hit_details` carry `seed` + `rank` per entry** (`:710-711`, `:733-734`) | **Yes — 7 files, all synthetic** (§2.6) |
| `diagnostics_outputs/tier_comparison.json` | `chapter_13_orchestrator.py:830-833` (inline `json.dump`); also `save_tier_comparison()` at `per_survivor_attribution.py:447-457` | tier → feature → mean attribution, plus `divergence` and `metadata` (`model_type`, `tier_sizes`, `total_survivors`, `attribution_attempts`, `attribution_failures`, `max_samples_per_tier`, `generated_at`, `version`) — `per_survivor_attribution.py:429-438` | **No — does not exist** (verified: `ls diagnostics_outputs/tier_comparison.json` → No such file) |
| `diagnostics_outputs/training_diagnostics.json` | `training_diagnostics.py:70` (`DEFAULT_OUTPUT_FILE = "training_diagnostics.json"`) | per-round snapshots incl. `feature_gradients.top_10` + `spread_ratio` (`:478-487`) — **aggregate, no identity** | Not present in the `diagnostics_outputs/` listing (only `compare_models_summary_S88_*.json` files and `history/`) |

Truncation caveat: `missed_details[:5]` / `hit_details[:5]` at `chapter_13_orchestrator.py:781-782` cap the archived per-survivor detail at 5 entries each — so even on a successful run, per-survivor records are capped well below the Top 20 actually examined (`:691`, `:718`). The aggregate `missed_relied_on` / `hits_relied_on` sets (`:743-751`, `:777-778`) are computed over **all** attributions, not the truncated 5.

Directory state at time of trace:
- `models/reinforcement/` — `best_model.cbm`, `best_model.json`, `best_model.meta.json`, `best_model.pth`, `best_model.txt` (all Mar 6), plus `compare_models/`, `tmp/`
- `predictions/` — `next_draw_prediction.json`, `history/` **only**
- `diagnostics_outputs/history/` — `root_cause_*.json` (7), plus `catboost_*.json`, `neural_net_*.json` diagnostics snapshots

## 2.5 Chapter 13 invocation trace

Imports at `chapter_13_orchestrator.py:57`: `from per_survivor_attribution import per_survivor_attribution, compare_pool_tiers`.

Live chain inside the diagnostic cycle (`Step 1b`, labelled *"Post-draw root cause analysis (observe-only, Ch14 Task 8.4)"* at `:321-323`):

```
:282   diagnostics = generate_diagnostics()
:290   save_diagnostics(diagnostics)
:325   hit_regression = self._detect_hit_regression(diagnostics)
:326   if hit_regression:
:328       predictions = self.load_predictions_from_disk(expected_draw_id=diagnostics.get("draw_id"))
:331       if predictions:
:335           model_info = self._load_best_model_if_available()
:336           if model_info:
:337               root_cause = self.post_draw_root_cause_analysis(
:338                   draw_result=diagnostics,
:339                   predictions=predictions,
:340                   model=model_info['model'],
:341                   model_type=model_info['model_type'],
:342                   feature_names=model_info.get('feature_names'),
:343               )
:345           else: logger.info("  Root cause skipped — no model available")
:347       else: logger.info("  Root cause skipped — no predictions available")
:349   if root_cause: result["steps"]["root_cause"] = {diagnosis, divergence, missed, hit}   # :350-355
:357   # Step 2: trigger evaluation proceeds independently
```

Inside `post_draw_root_cause_analysis()` (`chapter_13_orchestrator.py:652-811`):

```
:680   _feature_names = feature_names or getattr(self, 'feature_names', None)
:681-683   if not _feature_names: warn + return None            # ← Blocker 2 exits here
:685-687   if not predictions: warn + return None
:691   missed_top = [p for p in predictions[:20] if not p.get('hit')]
:693-695   if not missed_top: "all Top 20 hit — no analysis needed" + return None
:698-707   for survivor in missed_top: → per_survivor_attribution(...)   # GENUINE per-survivor
:708-715       append {'seed', 'rank', 'top_3_features'}
:718   hit_top = [p for p in predictions[:20] if p.get('hit')]
:721-730   for survivor in hit_top: → per_survivor_attribution(...)     # GENUINE per-survivor
:731-738       append {'seed', 'rank', 'top_3_features'}
:743-755   divergence_ratio = 1.0 - (len(overlap) / max(union_size, 1))
:757-764   diagnosis: 'training_issue' | 'regime_shift' (>0.5) | 'random_variance'
:766-783   build analysis dict
:785-791   log
:794-801   if diagnosis == 'regime_shift': self._run_regime_shift_analysis(...)
:804   self._archive_post_draw_analysis(analysis)
:805   return analysis
:807-811   except → log error + traceback + return None
```

And `_run_regime_shift_analysis()` (`:813-858`):

```
:827-829   tier_comparison = compare_pool_tiers(model, model_type, predictions, feature_names)   # TIER AGGREGATE
:830-833   write diagnostics_outputs/tier_comparison.json
:834       analysis['tier_comparison_path'] = tier_path
:837-838   if os.path.isfile("diagnostics_outputs/training_diagnostics.json"):
:840-845       StrategyAdvisor().request_diagnostics_analysis(diagnostics_path=..., tier_comparison_path=...)
:846-851       analysis['llm_analysis'] = ...
:852-855   except ImportError / Exception → non-fatal
```

Regression gate — `_detect_hit_regression()` at `chapter_13_orchestrator.py:536-561`: returns `True` on a `hit_rate` + `drop` substring match in `summary_flags` (`:547-550`), or on `hit_at_20 < previous_hit_at_20` when both are present (`:553-557`); otherwise `False` (`:559`), and `False` on any exception (`:560-561`). Described at `:541` as *"the gate that decides whether root cause analysis runs."*

**The call does reach a genuine per-survivor path** (`:702-707`, `:725-730` → `per_survivor_attribution.py:289-316` → single-sample backends). **Team Beta's `mean(dim=0)` finding does not apply to this path.** The `compare_pool_tiers` call at `:827` is the only aggregate step, and it is downstream and conditional.

### Preconditions that make it a no-op — two independent blockers

**Blocker 1 — the predictions file has no producer. `MISSING`.**

`load_predictions_from_disk()` defaults to `predictions/ranked_predictions.json` — `chapter_13_orchestrator.py:876-877` — and returns `None` if absent (`:894-896`).

A repo-wide search for `ranked_predictions` (excluding `backups/`) yields exactly **two** hits:
- `chapter_13_orchestrator.py:877` — the reader's default path
- `test_phase_8_soak.py:60` — `DEFAULT_PREDICTIONS_PATH = "predictions/ranked_predictions.json"`, a test constant

**No component writes this file.** Verified absent: `ls predictions/ranked_predictions.json` → No such file or directory. `predictions/` contains only `next_draw_prediction.json` (Mar 7) and `history/`.

Therefore `:331 if predictions:` is always false and the cycle logs `:347 "  Root cause skipped — no predictions available"`.

The reader imposes further requirements a future producer must satisfy:
- accepts a bare list or a dict with `predictions` / `ranked` key — `:901-916`
- staleness check: rejects if `data['draw_id'] != expected_draw_id` — `:904-910`
- rejects empty — `:918-920`
- **requires `'features'` in the first 5 records** — `:922-929`. So the producer must emit **full 91-feature vectors per prediction**, not just seeds and ranks.
- plus `'hit'`, `'seed'`, `'rank'` per the analysis function's own reads (`:691`, `:710-711`, `:718`, `:733-734`)

**Blocker 2 — `feature_names` is read from the wrong key. `MISSING`.**

`_load_best_model_if_available()` reads `meta.get("feature_names")` at **top level** — `chapter_13_orchestrator.py:582`, after loading `models/reinforcement/best_model.meta.json` (`:573`, `:578-579`).

Live inspection of that sidecar shows top-level keys are:

`['schema_version', 'model_type', 'checkpoint_path', 'checkpoint_format', 'feature_schema', 'signal_quality', 'data_context', 'training_metrics', 'hyperparameters', 'optuna', 'hardware', 'training_info', 'agent_metadata', 'provenance']`

**There is no top-level `feature_names`.** The names live at `feature_schema.feature_names` (89 entries, alongside `feature_count: 89`, `ordering: "lexicographic_by_key"`, `feature_schema_hash`, `excluded_features`, `source_file`). So `meta.get("feature_names")` → `None`.

The fallback at `chapter_13_orchestrator.py:623-635` then tries, in order:
- `model.feature_name()` — LightGBM (`:626-627`)
- `model.feature_names_in_` — sklearn-style (`:628-629`)
- `model.feature_names` — XGBoost (`:630-631`)

The current best model is `model_type: "neural_net"` (from the sidecar), loaded as a torch module at `:591-598` (`torch.load(..., map_location="cpu", weights_only=False)`, `model.eval()`). **A torch `nn.Module` has none of those three attributes.** So `feature_names` remains `None` (`:632` is not entered) and is returned as `None` in the dict at `:638-642`.

It is then passed to the analysis at `:342`, which exits immediately at `:680-683`:

```python
_feature_names = feature_names or getattr(self, 'feature_names', None)
if not _feature_names:
    logger.warning("post_draw_root_cause: no feature_names available, skipping")
    return None
```

`getattr(self, 'feature_names', None)` finds nothing — a grep of `chapter_13_orchestrator.py` for `feature_names` returns only `:342`, `:570`, `:582`, `:623-633`, `:641`, `:658`, `:674`, `:680-682`, `:706`, `:729`. **There is no `self.feature_names` assignment anywhere in the file** (no `__init__` binding).

Other model families would fare differently: xgboost (`:600-605`), lightgbm (`:607-611`), catboost (`:613-618`) — of these, LightGBM's `feature_name()` and XGBoost's `feature_names` could satisfy the fallback. So Blocker 2 is **specific to the currently-promoted neural_net model**, while Blocker 1 is universal.

**Both blockers are independent.** Fixing the missing predictions producer alone would still yield `None` for a neural_net best model (Blocker 2 exits at `:681-683`); fixing the sidecar key alone still has no predictions to read (Blocker 1 exits at `:347`). Blocker 2 is roughly a one-line fix (`meta.get("feature_schema", {}).get("feature_names")` at `:582`); Blocker 1 is a real integration gap requiring a Step 6 producer.

## 2.6 The archived artifacts are synthetic — provenance check

`diagnostics_outputs/history/root_cause_*.json` — **7 files**, dated Feb 14–15 2026. Newest is `root_cause_20260215_013533.json`, with keys exactly matching the schema built at `chapter_13_orchestrator.py:766-783`:

```json
{
  "type": "post_draw_root_cause",
  "draw_id": "2026-02-14_evening",
  "timestamp": "2026-02-15T01:35:33.154456+00:00",
  "missed_count": 3,
  "hit_count": 0,
  "attribution_success": {"missed": 3, "hit": 0},
  "feature_divergence_ratio": 1.0,
  "missed_relied_on": ["Column_27", "Column_36", "Column_42"],
  "hits_relied_on": [],
  "feature_overlap": [],
  "diagnosis": "training_issue",
  "missed_details": [
    {"seed": 123456, "rank": 1, "top_3_features": [["Column_42", 0.15473397547710588], ["Column_27", 0.15352718132518742], ["Column_36", 0.11142740529191185]]},
    {"seed": 789012, "rank": 2, "top_3_features": [["Column_42", 0.15473397547710588], ["Column_27", 0.15352718132518742], ["Column_36", 0.11142740529191185]]},
    {"seed": 345678, "rank": 3, "top_3_features": [["Column_42", 0.15473397547710588], ["Column_27", 0.15352718132518742], ["Column_36", 0.11142740529191185]]}
  ],
  "hit_details": []
}
```

Synthetic markers:
- `Column_27` / `Column_36` / `Column_42` are **not TFM feature names** — they are positional placeholders, absent from both the live 91 (§1.3) and the sidecar's 89
- seeds `123456` / `789012` / `345678` are obvious test values
- attribution vectors are **byte-identical across all three seeds** — consistent with a stub or untrained model over synthetic input
- `hit_count: 0` with `divergence_ratio: 1.0` → the `not hit_attributions` branch at `:758-760` yielding `diagnosis: 'training_issue'`

Provenance is the test harnesses, which call the real methods directly:
- `test_task_8_4.py` — *"Task 8.4 Smoke Test — post_draw_root_cause_analysis()"* (`:3`), calls at `:231`, `:261`, `:274`, `:285`; imports the module at `:44-45`; exercises `_detect_hit_regression` at `:75`, `:80`, `:85`, `:90`, `:94` and `load_predictions_from_disk` at `:109`, `:127`, `:141`, `:149`, `:162`
- `test_phase_8_soak.py` — real-method soak calling `_detect_hit_regression` (`:553`), `load_predictions_from_disk` (`:576`), `post_draw_root_cause_analysis` (`:595`); signature assertions at `:390-392`; and notably a recorded skip at `:442`: `detail="No model available -- cannot call post_draw_root_cause_analysis"`

**No artifact from a real draw exists.** The mechanism has never run on production data. The presence of these files is evidence the code path *executes correctly when fed data* — not evidence of production wiring.

## 2.7 Who consumes the output

**`root_cause_*.json` — nothing reads it.**

A grep for readers finds only the writer (`chapter_13_orchestrator.py:869-870`). The in-memory result is summarized into the cycle record at `:349-355` (`diagnosis`, `divergence`, `missed`, `hit`) and logged at `:785-791`. No trigger, no gate, no parameter decision consumes it. The code states this explicitly at `:321-323`: *"Runs if hit rate regression detected in diagnostics. Does NOT affect trigger evaluation — classification logged only."* Trigger evaluation proceeds independently at `:357-363`.

**`tier_comparison.json` — readers are wired, but the file never exists.**

- `diagnostics_llm_analyzer.py:215-226` — loads it if present (`:215` `if tier_comparison_path and os.path.isfile(...)`, `:217` open) and sets `'tier_comparison_available': True` (`:226`); parameter threaded through `:39`, `:114`, `:132` (documented as *"Path to tier_comparison.json (per-survivor attribution)"*), `:290`, `:310`, `:328`, `:340`, `:353`, `:375`
- `agents/watcher_agent.py:1747-1780` — passes it into `request_llm_diagnostics_analysis()`, but **conditionally**: `tier_comparison_path=(_tier_path if os.path.isfile(_tier_path) else None)` at `:1762-1764` and `:1776-1778`, with `_tier_path = 'diagnostics_outputs/tier_comparison.json'` at `:1752`. Since the file does not exist, **this is always `None`.** The whole block is additionally gated on `LLM_DIAGNOSTICS_AVAILABLE and health.get('action') == 'RETRY'` (`:1747-1748`, rationale `:1745-1746`), and requires `diagnostics_outputs/training_diagnostics.json` to exist (`:1754`)
- `chapter_13_orchestrator.py:840-851` — requests LLM analysis via `StrategyAdvisor.request_diagnostics_analysis()`, guarded by `ImportError` (`:852-853`) and reachable only when `diagnosis == 'regime_shift'` (`:794`, `:799-801`)
- historical patch scripts referencing the same wiring: `apply_s81_phase7_watcher_patch.py:275`, `:285`, `:299`

Downstream of the LLM: `agents/watcher_agent.py:1783-1790` applies `parameter_proposals` **only** through `_is_within_policy_bounds` clamping, and skips all proposals if that method is missing (`:1784-1787`). So even the one live consumer is advisory-with-clamp, not authoritative.

**`training_diagnostics` feature gradients — consumed as a health signal, not as attribution.**

`spread_ratio` is read at `training_diagnostics.py:538-539` and `:596-598`, checked against `SEVERITY_THRESHOLDS["gradient_spread_ratio"] = {"warning": 1000.0, "critical": 10000.0}` (`:77`). Related gradient-health thresholds: `gradient_norm_min` `{warning: 1e-6, critical: 1e-8}` (`:79`), used in the NN diagnosis at `:504`, `:524-528` off `self._gradient_norm_history` (`:349`, appended `:468-469`).

The `top_10` feature list written at `:479-485` is **never read by anything.**

**Net: no component anywhere acts on per-survivor attribution.** The single live consumer chain (tier comparison → LLM → clamped parameter proposals) is fed by a tier *aggregate*, and that file is never produced.

## 2.8 Classification

| Component | file:line | Classification |
|---|---|---|
| `per_survivor_attribution_nn` (`grad` / `grad_x_input`) | `per_survivor_attribution.py:49-127` | **IMPLEMENTED AND WIRED** — genuine single-sample (`:98`, `:108`); imported and called at `chapter_13_orchestrator.py:57`, `:702`, `:725`. Unreachable in production due to Blockers 1 & 2. |
| `per_survivor_attribution_xgb` (`pred_contribs`) | `per_survivor_attribution.py:134-177` | **IMPLEMENTED AND WIRED** — same reachability caveat |
| `per_survivor_attribution_lgb` (`pred_contrib`) | `per_survivor_attribution.py:184-224` | **IMPLEMENTED AND WIRED** — same caveat |
| `per_survivor_attribution_catboost` (native SHAP) | `per_survivor_attribution.py:231-273` | **IMPLEMENTED AND WIRED** — same caveat |
| `per_survivor_attribution()` dispatcher | `per_survivor_attribution.py:289-316`, table `:281-286` | **IMPLEMENTED AND WIRED** |
| `post_draw_root_cause_analysis()` — per-survivor with seed identity | `chapter_13_orchestrator.py:652-811`; identity `:710-711`, `:733-734`; archive `:804`, `:860-873` | **IMPLEMENTED BUT NOT CONSUMED** — writes a seed-indexed artifact; zero downstream readers; self-declared observe-only (`:321-323`) |
| `_detect_hit_regression()` gate | `chapter_13_orchestrator.py:536-561`, called `:325` | **IMPLEMENTED AND WIRED** |
| `_load_best_model_if_available()` (4 model families) | `chapter_13_orchestrator.py:563-646` | **IMPLEMENTED AND WIRED** — loads the model successfully; returns `feature_names: None` for neural_net (Blocker 2) |
| `load_predictions_from_disk()` reader | `chapter_13_orchestrator.py:875-937`, called `:328` | **IMPLEMENTED AND WIRED** — always returns `None` in production (no input file) |
| `predictions/ranked_predictions.json` **producer** | — (only `chapter_13_orchestrator.py:877` reader, `test_phase_8_soak.py:60` constant) | **MISSING** — no writer in repo; makes the whole path a permanent no-op |
| `feature_names` at sidecar top level | read `chapter_13_orchestrator.py:582`; actual location `feature_schema.feature_names` | **MISSING** — wrong key read; second independent no-op, neural_net-specific |
| `self.feature_names` on the orchestrator | fallback at `chapter_13_orchestrator.py:680` | **MISSING** — never assigned anywhere in the file |
| `compare_pool_tiers()` — tier aggregate | `per_survivor_attribution.py:323-440`; reduction `:399-402` | **IMPLEMENTED BUT NOT CONSUMED** — averages away survivor identity; file never produced, so LLM readers always receive `None` (`agents/watcher_agent.py:1762-1764`, `:1776-1778`) |
| `save_tier_comparison()` | `per_survivor_attribution.py:447-457` | **IMPLEMENTED BUT NOT CONSUMED** — orchestrator inlines its own `json.dump` at `chapter_13_orchestrator.py:832-833` instead of calling it |
| `_run_regime_shift_analysis()` | `chapter_13_orchestrator.py:813-858` | **IMPLEMENTED BUT NOT CONSUMED** — reachable only on `regime_shift` (`:794`), itself downstream of both blockers |
| `_archive_post_draw_analysis()` | `chapter_13_orchestrator.py:860-873` | **IMPLEMENTED BUT NOT CONSUMED** — 7 artifacts written, all synthetic; no reader |
| `training_diagnostics` feature gradients | `training_diagnostics.py:474`; hook `:429-431`; output `:478-487` | **AGGREGATE, NOT PER-SURVIVOR** — TB confirmed. As *batch* importance: wired and consumed via `spread_ratio` (`:538`, `:596`, thresholds `:77`). As *per-survivor* attribution: **MISSING** — no identity is even available to it. `top_10` output has no reader. |
| Chapter 11 `feature_importance` (native / permutation / gradient) | `feature_importance.py:180-205`; gradient `:465-499`; permutation `:362`, `:405`; native `:191-192` | **IMPLEMENTED AND WIRED, aggregate by design** — global importance, correctly labelled, not per-survivor |
| `feature_importance.py` canonical feature list | `feature_importance.py:95-119` (46 + 14 = 60 names); duplicated `feature_drift_tracker.py:206-207` | **STALE** — 31 live features absent (23 `batt_*`, 4 digit, 4 metadata); see §1.5 |
| Any consumer acting on per-survivor attribution | — | **MISSING** — observational only, end to end |

## 2.9 Real status, stated plainly

The per-survivor attribution *engine* is real, correct, and better than Team Beta's review suggests. TB read `training_diagnostics.py:474`, which is a genuinely batch-averaged path with no survivor identity available to it — that finding is exactly right about that file. But `per_survivor_attribution.py` is a **separate module** implementing true single-sample attribution across four model families (`:49`, `:134`, `:184`, `:231`), with prior TB corrections already applied (`:83-88`, `:93-95`), and Chapter 13 invokes it with seed identity attached (`chapter_13_orchestrator.py:702`, `:710`, `:725`, `:733`).

What is missing is the *plumbing around it*:

1. the predictions artifact it reads does not exist and has no producer (§2.5 Blocker 1)
2. the feature-names lookup reads a key absent from the sidecar (§2.5 Blocker 2)
3. its output has no consumer (§2.7)
4. every artifact on disk is synthetic (§2.6)

The correct summary is: **implemented and invoked, never executed on real data, and observational by design even if it were.** Not "designed-documented only" — the code is real and test-exercised. Not "implemented and wired" at the system level — it cannot run.

Two cheap, high-value fixes if this is to be made live:

1. point `chapter_13_orchestrator.py:582` at `meta.get("feature_schema", {}).get("feature_names")` — one line, unblocks the neural_net case
2. have Step 6 write `predictions/ranked_predictions.json` including per-prediction `features` (full 91-vector), `seed`, `rank`, `hit`, and `draw_id` — satisfies the reader's schema check at `:922-929` and the staleness check at `:904-910`

Neither is in scope for this trace — flagged only.

---

# Appendix A — Cross-question link

Question 1 and Question 2 intersect at the feature-name contract:

- Live extraction emits **91** features (`full_scoring_worker.py:698-699`; verified on `full_scoring_results/full_scoring_results_20260311_170236/chunk_0000.json`)
- The trained model was fitted on **89** (`models/reinforcement/best_model.meta.json` → `feature_schema.feature_count`), excluding `score`, `confidence`, `holdout_hits`, `holdout_quality`
- The Chapter 11 canonical list declares **60** (`feature_importance.py:95-119`) — the likely source of the docs' "~62"
- Attribution requires `feature_names` to be **positionally aligned** with the feature vector it is given (`per_survivor_attribution.py:117`, `:173`, `:220`, `:269` all `enumerate(feature_names)` against raw index order)

So if Blocker 2 is fixed by wiring `feature_schema.feature_names` (89, `ordering: lexicographic_by_key`) into attribution, the predictions producer built for Blocker 1 must emit feature vectors in **exactly that 89-name lexicographic order with `score` and `confidence` excluded** — not the raw 91-key dict order from `full_scoring_worker.py:393-394`. Mismatched ordering would produce silently wrong attributions rather than an error, since every backend zips names against positions without validation. The synthetic `Column_27` / `Column_42` artifacts in §2.6 are a preview of exactly this failure mode.

# Appendix B — Verification methods used

- **Source reads** — `convert_survivors_to_binary.py`, `generate_step3_scoring_jobs.py`, `full_scoring_worker.py`, `survivor_scorer.py`, `models/global_state_tracker.py`, `utils/survivor_loader.py`, `feature_importance.py`, `training_diagnostics.py`, `per_survivor_attribution.py`, `chapter_13_orchestrator.py`, `agents/watcher_agent.py`, `diagnostics_llm_analyzer.py`
- **Repo-wide searches** with `/bin/grep` (not the shell `grep` wrapper, which honors `.gitignore` and skips `*.json`) for: the 5 dead feature names, `ranked_predictions`, `per_survivor_attribution` / `compare_pool_tiers` / `save_tier_comparison` / `tier_comparison`, `post_draw_root_cause_analysis` / `_detect_hit_regression` / `load_predictions_from_disk` / `_load_model_for_root_cause`, `mean(dim=0)` / `mean(axis=0)`, `feature_names`. `backups/` and `step6_restoration/` excluded from producer searches (historical copies, not live code).
- **Live artifact inspection** (read-only, `json.load` in an ephemeral `python3 -c`): latest Step 3 chunk output — feature count, key list, per-feature unique-value counts across 84 records; `models/reinforcement/best_model.meta.json` — top-level keys, `feature_schema` contents, set-diff against the live vector; `feature_importance.py` constants parsed and set-diffed against the live vector; newest `diagnostics_outputs/history/root_cause_*.json` — schema and provenance markers
- **Filesystem existence checks** — `models/reinforcement/`, `predictions/`, `diagnostics_outputs/`, `diagnostics_outputs/history/`, and explicit negative checks on `predictions/ranked_predictions.json` and `diagnostics_outputs/tier_comparison.json`

**No files were modified. No pipeline was launched. No commits or pushes were made.**
