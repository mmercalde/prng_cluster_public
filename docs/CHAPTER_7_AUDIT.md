# Chapter 7 audit — Prediction Generator (Step 6) vs. source

**Repo:** `/home/michael/distributed_prng_analysis` on VM 101, user `michael`. Code base
`a15b5247336255801de8cf67d52653701b1e8c80` (`a15b524`).
**Code moved since `a15b524`, once:** current HEAD `b6cb789` ("watcher: failure authority, dry-run refusal, single
halt authority (Ch12 B1, B6, B3)") changed exactly `agents/watcher_agent.py`, `agents/watcher_dispatch.py` and
`tests/test_watcher_safety_repairs.py`. Every WATCHER anchor below is therefore read from the committed blob
(`git show a15b524:agents/watcher_agent.py`) and is marked `wa@`. Every other cited file is byte-identical at
`a15b524` and HEAD (`git diff --quiet a15b524 -- prediction_generator.py models/ agent_manifests/ survivor_scorer.py
integration/metadata_writer.py utils/canonical_arrays.py window_optimizer_integration_final.py
docs/PIPELINE_BEHAVIOUR_MODEL.md` → exit 0).
**Date:** 2026-11-11. Read-only. Nothing was predicted, trained, run or dispatched. "Works" below means the code path
is complete on reading; host artifacts are cited where they corroborate.
**Audited:** `docs/CHAPTER_7_PREDICTION_GENERATOR.md` (933 lines at `a15b524`).
**Numbering:** bare "Step N" = the **executable** scheme (`docs/PIPELINE_BEHAVIOUR_MODEL.md` §1.1). Step 6 =
`prediction_generator.py`: `wa@:394` `STEP_SCRIPTS[6]`, `wa@:405` `STEP_MANIFESTS[6] = "prediction.json"`,
`agent_manifests/prediction.json:4` `"pipeline_step": 6`, PBM §1.3 row 6. The chapter title uses the same scheme;
its §1.3 "Step 2.5" is the old name of executable Step 2 (D3).
**Framing held throughout:** TFM is functional mimicry of PRNG surface output. A survivor is a scored candidate; the
top candidates vote their next value into a pool; the target is a pool hit rate. Nothing here assumes a "true seed".

Abbreviations: `pg` = `prediction_generator.py`; `mf` = `models/model_factory.py`; `fs` = `models/feature_schema.py`;
`ss` = `survivor_scorer.py`; `nnw` = `models/wrappers/neural_net_wrapper.py`; `mw` = `integration/metadata_writer.py`;
`pm` = `agent_manifests/prediction.json`; `WOIF` = `window_optimizer_integration_final.py`; `wa@` =
`agents/watcher_agent.py` at `a15b524`; `PBM` = `docs/PIPELINE_BEHAVIOUR_MODEL.md`; `CH7` = the chapter.

---

## 1. Verdict

```
ACCURATE                 18 claims
BROKEN                    3 claims   (1 already recorded in the governance trail: B3)
SWITCHED-OFF-BY-RULING    1 claim
DESCRIPTION-ONLY         16 claims
UNVERIFIED                1 claim
Absorbed into an item     8 claims   (counted once, inside the item named in §7)
---------------------------------------
Total checked            47 claims
```

Null results first, each with its denominator:

1. **Step 6 has produced 0 predictions since its survivor input became a count-only stub.** Step 1 has written
   `forward_survivors.json` / `reverse_survivors.json` as `{"survivor_count": N, "note": ...}` since S166-ACCUM
   (`8ebdf99`, 2026-04-21; today `WOIF:2945-2953`). On that input the default (dual-sieve) path raises `ValueError`
   and the forward-only path writes nothing. Host: 19 of 19 files in `predictions/history/` are dated 2026-02-08 to
   2026-03-07; 0 are later (B1).
2. **0 of 1 Step-6 model loads compare the feature-schema hash** — already Chapter 6 B5, cross-referenced (X1).
3. **0 of the top-`pool_size` votes are aggregated.** Each top survivor contributes one value; repeats are kept as
   separate entries, with no weighting and no per-value count. Host: 4 of 19 history files hold a repeated value
   among their 20 (B3, governed by `TB_RULING_S179_IMPLEMENTATION_AUTH.md` §2.5).
4. **0 of the chapter's 3 output files other than `next_draw_prediction.json`, and 0 of its 3 pools, exist in code**
   — already PBM D3 / Chapter 13 B1 (X3).
5. **A missing checkpoint does not fail Step 6.** It is caught, and survivors are ranked by the plain mean of their
   feature row, reported as `status: success`, exit 0 (B2).

Short version: the sidecar-only loading contract and the signal-quality gate are real and work as written. What the
chapter calls the pool — three tiers, weighted voting, ranked and pool files — was never built. And the live
survivor input has been a stub since 2026-04-21, nearly seven months.

---

## 2. The chapter-specific questions

### Q1. Which step is this, and in which scheme?

Executable Step 6 (see header). The chapter's title, §1.1 and §10.1 use the executable scheme. §1.3 "Step 2.5" and
the end-of-chapter table use old names and wrong chapter numbers (D3, D16).

### Q2. How do the top candidates vote values into the pool? (anchored)

Read at `a15b524`, in execution order:

1. **Survivors in.** `main` reads `--survivors-forward` and, if given, `--survivors-reverse` as a list or as
   `data.get('survivors', data)` (`pg:1042-1050`). In dual-sieve mode (`use_dual_sieve` default True, `pg:112`;
   WATCHER always passes a reverse file, `pm:179`) the working set is `ss.compute_dual_sieve_intersection`
   (`pg:606-616`). That function coerces both lists to bare integer seeds (`ss:128-137`) and returns
   `sorted(list(intersection))` (`ss:172`), so every survivor that reaches the pool is an `int`.
2. **Features.** For an `int` survivor the features are recomputed now:
   `scorer.extract_ml_features(seed, lottery_history, skip=default_skip)` (`pg:785-790`). That compares the seed's
   regenerated sequence with the history from index 0 (`ss:354-365`). For a `dict` survivor the row comes from
   `survivor["features"]` (`pg:782-784`). The row is ordered by the sidecar's
   `feature_schema.per_seed_feature_names` / `feature_names` (`pg:366`), and any missing name becomes 0.0 (`pg:798`;
   Ch6 B5).
3. **Scaling.** For `neural_net` only, the checkpoint's stored scaler is applied (`pg:814-824`, loaded at
   `pg:385-450`).
4. **Rank.** `predicted_quality = self.model.predict(X)` (`pg:828`). Order is `np.argsort(predicted_quality)[::-1]`
   (`pg:836`). There is no tiebreak rule (S179 §2.5 requires one).
5. **Vote.** For each of the first `pool_size` ranked survivors (`pg:841`), the seed's sequence is regenerated to
   length `len(lottery_history)+1` with that survivor's skip (`pg:846`). The value at index `len(lottery_history)`
   (`pg:839`, `:849`) is its vote. `_generate_sequence` reduces by `mod` (`ss:119-125`). The vote carries the
   survivor's raw quality as `confidence_scores` (`pg:850`). **The skip is the survivor's own only when the survivor
   arrives as a dict** (`pg:530-553`). In the dual-sieve path every survivor is an int, so every vote uses
   `default_skip` (0 by default, `pg:109`) — see N1.
6. **Pool.** The list of votes is kept in rank order, **one entry per survivor, repeats included**
   (`pg:848-850`). It is truncated to the first `k` (`pg:653-664`). Confidences are a sigmoid of each raw score's
   z-score within those `k`, or 0.5 for all when the scores are constant (`pg:686-701`).
7. **Out.** `{"status": "success", "predictions": [int,…], "raw_scores": […], "confidence_scores": […],
   "confidence_scores_normalized": […], "metadata": {…}}` (`pg:709-733`), plus `agent_metadata` (`pg:929-942`).
   It is written to `predictions/next_draw_prediction.json` and a timestamped copy in `predictions/history/`
   (`pg:905-913`, `:944-952`). The output records no seed, rank or skip.

Under WATCHER at `a15b524` the command is built from `pm:176-184`. `pm:32` maps CLI `--k` to the parameter
`pool_size`, so both `k` and `pool_size` are emitted as `--k 20` and `--pool-size` is never sent (`wa@:2104-2154`).
`pool_size` therefore stays at its default 20 (`pg:89`). Host logs from before that mapping
(`logs/s120_step4_6.log:1043`, 2026-03-06) show `--k 20 --pool-size 20`.

### Q3. What changed because of RANGE-MINER, the 22-array NPZ, or the window anchor?

- **Survivor input (B1).** The canonical carrier from Step 1 is the certified 22-array NPZ generation
  (`utils/canonical_arrays.py:98` `CANONICAL_ARRAY_CONTRACT`; finalizer call in `WOIF` after `:2953`). Steps 2–3
  consume it (`PROPOSAL_S172_RANGE_MINER_v1_4_4.md` §4.4.1). Step 6 does not read the NPZ or Step 3's
  `survivors_with_scores.json`. It reads the forward/reverse JSON files, which became count-only stubs at S166 and
  stayed stubs under the D3.5 finalizer (`46a3828`, `WOIF:2945-2953`). The S172 proposal lists this as "Channel D"
  and says "runtime tolerates summary stub" (`PROPOSAL_S172_RANGE_MINER_v1_4_4.md:249`, `:320`). The code does not
  tolerate it.
- **Dataset path.** P0.5 dataset authority rewrites Step 6's `lottery_history` parameter from bare `daily3.json` to
  the run's frozen absolute immutable path before dispatch (`wa@:474` `_P05_DATASET_PARAM_KEYS`, `wa@:1862-1870`).
- **Window anchor / generator phase (Brief I).** `pg` and `ss` contain neither `window_anchor` nor
  `generator_phase` (`/bin/grep -n` → no hit). Step 6 reads the vote at generator output index `len(history)` with
  generator phase 0 aligned to history index 0 (`pg:839`, `:846`; same alignment as `ss:354-365`). No Brief I
  document mentions Step 6 or `prediction_generator` (`/bin/grep` over `docs/*WINDOW_ANCHOR*`, `docs/*BRIEF_I*`,
  `docs/TB_RULING_WINDOW_ANCHOR*` → no hit). Whether that alignment is right for a survivor found at
  `window_anchor ≠ 0` is not settled anywhere. Open item O3, not counted.
- **RANGE-MINER execution.** None. Step 6 is not distributed (`pm` `"distributed": false`); Chapter 9 audit row 22
  agrees.

### Q4. The §5 host question for all chapters — `agent_manifests/definitions.json`

Read on the host. Gitignored (`git check-ignore -v` → `.gitignore:41:*.json`), 220 lines.
`pipeline_steps.step6_prediction` (`:60-96`) declares version 6.0.0, consumes `signal_quality` (gate on
`prediction_allowed`), `data_context` and `feature_schema` ("validate": "feature_schema_hash"). Its outputs are
`predictions/next_draw_prediction.json`, `predictions/prediction_pools.json` and `predictions/ranked_survivors.json`
(`:80-84`). On schema mismatch it says `fatal_error` (`:91-94`). **No tracked or untracked `*.py` / `*.sh` reads it**
(`/bin/grep -rln "definitions\.json" --include=*.py --include=*.sh .` outside `backups/` → none). Of its three Step 6
outputs only the first exists on the host. Nothing in the tree writes the other two (`/bin/grep -rln
"ranked_survivors.json\|prediction_pools.json"` over `*.py`/`*.sh` → `build_pools.py`, `evaluate_pools.py`,
`test_reinforcement_with_real_survivors.py`, none of them Step 6). This adds no new label: its `validate` line is
Ch6 B5, and its output names are X3.

---

## 3. What actually works end to end today (reading + host artifacts)

1. **Sidecar-first loading, type from the sidecar only.** `pg:290-297` reads `best_model.meta.json`.
   `load_model_from_sidecar` takes `model_type` from it (`mf:308`) and calls `load_model(model_type, path)`
   (`mf:374`, `mf:98`). The extension is never consulted. A missing sidecar is fatal (`pg:292-294`, `mf:291-296`);
   a missing `model_type` or `checkpoint_path` is fatal (`mf:302-315`).
2. **Signal-quality gate before model load.** If `signal_quality.prediction_allowed` is false, Step 6 skips the
   model load (`pg:308-342`), writes a `PREDICTION_SKIPPED` record to the canonical path (`pg:580-587`) and exits 3
   (`pg:1074-1084`). A second check, `_check_signal_gate`, repeats this for the lowercase `prediction_skipped`
   status (`pg:465-524`, `:593-597`). Host sidecar: `signal_status "weak"`, `prediction_allowed true`.
3. **NN input scaling** from the checkpoint, gated to `neural_net` (`pg:361-362`, `:814-824`).
4. **Ranking and one value per top survivor** as in Q2.
5. **Canonical output and lineage.** `predictions/next_draw_prediction.json` (`pg:905-907`) with `parent_run_id`
   taken from the Step 5 sidecar's `agent_metadata.run_id` (`pg:302-303`), `pipeline_step 6`,
   `follow_up_agent None` (`pg:936-938`, `mw:156-158`). Host file (2026-03-07 19:08): `parent_run_id
   step5_20260306_212952`, matching the sidecar's `provenance.run_id`.
6. **Terminal step.** `wa@:1390-1393` declares the pipeline complete after Step 6; `pm:39` `follow_up_agents: []`.

Host corroboration (read-only; keys and mtimes). `predictions/` holds `next_draw_prediction.json` (1,483 B,
2026-03-07) and `history/` (19 files, 2026-02-08 to 2026-03-07). All 19 have `metadata.method = dual_sieve`,
`k = 20`, `pool_size = 20`. Prediction counts: 18 × 20 and 1 × 2. Feature counts: 62 (15 files) and 89 (4 files).
Model types: catboost, lightgbm, neural_net. `forward_survivors.json` and `reverse_survivors.json` are 126-byte
stubs dated 2026-08-21. `optimal_window_config.json` and `reinforcement_engine_config.json` do not exist in the
working tree today.

---

## 4. BROKEN

"Governed" means a ruling or governed document already records the condition. That is a status, not a new finding.

**B1 — Step 6's survivor input has been a count-only stub since S166, and Step 6 cannot consume it.**
*(not found in the governance trail as a defect. The S172 proposal asserts the opposite: "runtime tolerates summary
stub", `PROPOSAL_S172_RANGE_MINER_v1_4_4.md:249`)*
- Chapter: inputs `forward_survivors.json (from Step 1)` (§2.1, §2.2); "Extract features for forward survivors"
  (§2.1); every usage example passes `--survivors-forward forward_survivors.json` (§9.2, §11.2).
- Code: WATCHER passes `forward_survivors.json` and `reverse_survivors.json` (`pm:178-179`). Step 1 writes both as
  `{"survivor_count": N, "note": "Full survivors omitted — objects not retained; see …"}` (`WOIF:2945-2953`, since
  `8ebdf99` 2026-04-21 and kept by `46a3828` 2026-07-25). In `--config-file` mode they are written as `[]` (Chapter 1
  audit C-5). `main` keeps the dict, because it has no `survivors` key (`pg:1044`, `:1050`). In dual-sieve mode
  `_coerce_seed_list` iterates the dict's keys and calls `int("survivor_count")` (`ss:136`), which raises
  `ValueError`; it is uncaught, so Step 6 exits 1. In forward-only mode, or with `[]`, every entry is skipped
  (`pg:791-793`), the pool is empty, and `generate_predictions` returns before `_save_predictions` (`pg:631-640`).
  That is exit 0 with no new file, so the March 2026 `next_draw_prediction.json` stays in place.
- Step 6 never reads the population Step 5 trained on. The manifest lists `survivors_with_scores.json` as required
  (`pm:8`), and `_save_predictions` writes it into `agent_metadata.inputs` (`pg:933`), but no code in `pg` opens it.
  Host: the sidecar's model was trained on 85 survivors from `survivors_with_scores.json`; the last Step 6 run
  ranked a 2-seed forward∩reverse intersection of a different file pair.
- Host: 0 of 19 Step-6 outputs postdate 2026-04-21. Today WATCHER would stop Step 6 earlier anyway, with a HARD
  freshness failure, because `optimal_window_config.json` (`pm:10`) is missing (`wa@:816-818`). That is a
  point-in-time host state, not a code path.
- Stops at: `ss:136`.
- Repair would have to: give Step 6 the survivor population Step 3 scored and Step 5 trained on (for example, the
  certified NPZ generation or `survivors_with_scores.json`), with the per-survivor fields the vote needs, and make a
  stub or empty input a loud failure rather than an exception or a silent no-write.

**B2 — A missing checkpoint does not fail Step 6; it ranks by the mean of the feature row instead.**
*(not found in the governance trail: `/bin/grep -rln "fallback_mean\|mean fallback" docs/` → none; the only other
occurrence is the patch draft `step6_restoration/prediction_generator_PATCH.py:311`)*
- Chapter: `load_best_model` "Fails fast on any validation error" and raises `FileNotFoundError` (§3.2); the
  troubleshooting table assumes a fatal stop (§11.1).
- Code: `load_model_from_sidecar` raises `FileNotFoundError` when the checkpoint is absent (`mf:319-330`).
  `PredictionGenerator.__init__` catches `FileNotFoundError` and logs a warning (`pg:375-376`), leaving
  `self.model = None`. `_build_prediction_pool` then scores with `np.mean(X, axis=1)` (`pg:827-833`). The result is
  written as `status: success` with `model_type: fallback_mean` (`pg:709-733`), and the process exits 0. The
  docstring's exit code 2 "No model found" (`pg:32`) is never returned (`/bin/grep -n "return 2" pg` → none).
- Stops at: `pg:375`.
- Repair would have to: make a missing checkpoint (with `prediction_allowed` true) a non-zero exit with a named
  reason, or record a governed decision that the mean fallback is intended and label its output as not model-based.

**B3 — The pool is not a vote: no aggregation, no weighting, repeats kept, truncated to `k`.**
*(GOVERNED: `docs/TB_RULING_S179_IMPLEMENTATION_AUTH.md` §2.5 — "The current generator ranks predictions and then
truncates to `pool_size`, so it does not currently preserve a full deterministic unique ranking". The ruling
approves a ranked unique artifact (raw_rank, unique_rank, predicted outcome, quality, stable survivor id, source
index, duplicate-of). Sequence item 4 at `:540`. Not implemented at `a15b524`: Chapter 13 audit A.10 places S179's
implementation as "did not land".)*
- Chapter: `build_prediction_pools` adds each survivor's `predicted_quality` to `votes[next_pred]`, normalises per
  value, sorts by weight and reports `unique_predictions` / `top_prediction` (§6.2); `generate_next_draw_prediction`
  combines pools with weights 0.5/0.3/0.2 (§7.1); "Weighted voting aggregates predictions" (§13).
- Code: one value per ranked survivor, appended in rank order (`pg:841-850`), first `k` kept (`pg:653-664`). There
  is no per-value sum, no normalisation, no unique count and no cross-pool combination. `git log --all
  -S"build_prediction_pools"` finds only documentation commits (`53a3829`, `3b9286c`, `d8b52ab`), so the mechanism
  was never in code. Host: 4 of 19 history files contain a repeated value among 20 entries (19 unique).
- Stops at: `pg:849`.
- Repair would have to: what S179 §2.5 already specifies (deterministic order, duplicate collapse, a persisted
  survivor id), plus whatever weighting or tiering the owner chooses. The chapter's 20/100/300 tiers are a separate
  owner decision (X3).

*(The per-survivor-skip finding is not a chapter claim; it is recorded as N1 in §7 and carries no label.)*

---

## 5. SWITCHED-OFF-BY-RULING

**S1 — `--output-dir` is accepted and ignored; the output path is fixed.**
- Ruling: the Team Beta ruling of 2026-01-28 cited at `pg:902` ("Per Team Beta ruling (2026-01-28): WATCHER only
  validates canonical path") and by commit `c6e3989` ("single canonical output, optional archive"). **The ruling text
  itself was not found as a file:** searched `docs/TB_RULING_*`, `docs/SESSION_CHANGELOG_202601*`, and
  `/bin/grep -rln next_draw_prediction docs/` (the hits restate the path; none is the ruling). The label rests on
  the in-code citation and the commit.
- Chapter: `--output-dir DIR  Output directory (default: predictions/)` (§9.1), used in §9.2.
- Code: `--output-dir` sets `config.predictions_dir` (`pg:1022-1023`), which nothing reads. `_save_predictions`
  always writes `Path("predictions")` relative to the working directory (`pg:905-907`).
- If lifted: Step 6 could write to the directory it is given. WATCHER's success condition and freshness check name
  `predictions/next_draw_prediction.json` only (`pm:12`, `pm:40-41`; `wa@:803-834`), so a run sent elsewhere would
  look to WATCHER like a missing or stale output unless the manifest moved with it.

---

## 6. DESCRIPTION-ONLY

| # | Chapter says | Code at `a15b524` | Anchor |
|---|---|---|---|
| D1 | Header "Version 1.0", "Lines ~400"; §13 component line counts | 1,095 lines; module says v6.1 (`pg:15`), CLI/dry-run print v6.0 (`pg:962`, `:1033`) | `wc -l`; `pg:15` |
| D2 | "Works with all 4 model types" (§1.2) | 5: `neural_net`, `xgboost`, `lightgbm`, `catboost`, `random_forest` | `mf:25`, `mf:54-67` |
| D3 | §1.3 position: "Step 2.5: Scorer Meta-Optimizer"; no Step 0 | Executable Step 2 = `run_scorer_meta_optimizer.sh`; Step 0 = `trse_step0.py` | `wa@:387-395` |
| D4 | Input `lottery_history.json (recent draws)` (§2.1, §2.2) | WATCHER passes `daily3.json` (`pm:180`), rewritten by P0.5 to the frozen absolute dataset path (`wa@:1862-1870`). `main` accepts `[int]`, `{"draws": …}` or `[{"draw": …}]` (`pg:1052-1056`) | as cited |
| D5 | `load_best_model(models_dir, survivors_file)` calling `ModelFactory.load` (§3.2) | `load_model_from_sidecar(models_dir, device, survivors_file=None, strict=True)` → `load_model(model_type, path, device)`; checkpoint path absolute, with a fallback to `models_dir/<name>` | `mf:262`, `mf:98`, `mf:319-330`, `mf:374` |
| D6 | Sidecar example: `schema_version 3.1.2`, relative `checkpoint_path`, `feature_count 50`, `validation_metrics` (§3.3) | Host sidecar top-level keys: `schema_version, model_type, checkpoint_path` (absolute), `checkpoint_format, feature_schema, signal_quality, data_context, training_metrics, hyperparameters, optuna, hardware, training_info, agent_metadata, provenance`; 89 features, 4 excluded | host `models/reinforcement/best_model.meta.json` |
| D7 | `rank_survivors(model, survivors, feature_names)` attaches `predicted_quality` and `rank` (§5.1) | No such function; ranking is inline in `_build_prediction_pool`. Int survivors get features from `extract_ml_features`; no `rank` field is written | `pg:752-862`, `:785-790`, `:836` |
| D8 | `scorer.predict_next(seed, lottery_history)` (§6.2) | `scorer._generate_sequence(seed, len(history)+1, skip)[len(history)]` | `pg:839-849`; `ss:119-125` |
| D9 | `next_draw_prediction.json` holds `top_predictions`, `primary_prediction`, `coverage`, `pool_contributions` (§7.2) | Holds `status, predictions[int], raw_scores, confidence_scores, confidence_scores_normalized, metadata, agent_metadata` (host file agrees) | `pg:709-733`, `:929-946` |
| D10 | CLI: `--models-dir` required; `--pool-sizes`, `--prng-type`, `--mod` (§9.1); `--verbose` (§11.2) | `--models-dir` optional (default `models/reinforcement`). None of the four other flags exists. Actual: `--config, --survivors-forward, --survivors-reverse, --lottery-history, --models-dir, --parent-run-id, --no-require-schema, --k, --pool-size, --output-dir, --log-level, --test, --dry-run`. PRNG type and mod come from `--config` `prng` or `optimal_window_config.json` | `pg:965-986`, `:150-165` |
| D11 | Manifest shown with `agent_id`, `step_id`, `inputs` objects, three `outputs`, `success_criteria` (§10.1) | `agent_name, pipeline_step 6, version 1.5.0, required_inputs` (4), `primary_output`, `outputs` (1), `actions[0].args_map`, `default_params`, `evaluation_type: file_exists` | `pm:2-41`, `:176-196` |
| D12 | `agent_metadata` with `pipeline_step_name 'prediction_generation'`, `run_id`, `model_type`, `model_r2` (§10.3) | `inject_agent_metadata` writes `inputs, outputs, parent_run_id, pipeline_step 6, pipeline_step_name "prediction", follow_up_agent null, confidence, reasoning, timestamp_injected` | `pg:929-942`; `mw:47`, `:127-158`; host file |
| D13 | Troubleshooting messages (§11.1): "Missing metadata sidecar", "Empty predictions → check input" | Step 6 raises its own `Required sidecar missing` first (`pg:294`). An empty pool returns a no-predictions dict without writing a file (`pg:631-640`). "Feature schema mismatch" is never raised (X1) | as cited |
| D14 | §11.3 verify the hash against `forward_survivors.json` | The training hash comes from `survivors_with_scores.json` (sidecar `feature_schema.source_file`). On today's stub, `get_feature_schema_from_data` has no record to read | `fs:162-188`; host sidecar |
| D15 | §12 method reference (`load_best_model`, `rank_survivors`, `extract_features_ordered`, `build_prediction_pools`, `aggregate_votes`, `generate_next_draw_prediction`, `save_*`) | Actual: `PredictionConfig.from_json`, `PredictionGenerator.__init__`, `_load_nn_scaler`, `_check_signal_gate`, `_get_survivor_skip`, `generate_predictions`, `_build_prediction_pool`, `extract_features_batch` (no caller), `_save_predictions`; `fs.get_feature_schema_with_hash` and `fs.validate_feature_schema_hash` exist, and the latter is imported and never called | `pg:81-952`; `fs:162`, `fs:193` |
| D16 | §13 calls itself "Chapter 16"; "Pipeline Complete" table maps Step 2.5→Ch 13 `scorer_trial_worker.py`, 3→Ch 6/8, 4→Ch 4 `meta_prediction_optimizer.py`, 5→Ch 4/10, 6→Ch 16 | Step 0 `trse_step0.py`; 1 `window_optimizer.py` (Ch 1, sieve inside it Ch 2); 2 `run_scorer_meta_optimizer.sh` (Ch 3); 3 `run_step3_full_scoring.sh` (Ch 4); 4 `adaptive_meta_optimizer.py` (Ch 5); 5 `meta_prediction_optimizer_anti_overfit.py` (Ch 6); 6 `prediction_generator.py` (Ch 7) | `wa@:387-395`; `docs/CLAUDE_CODE_INSTRUCTIONS_CHAPTER_CORRECTIONS_20261111.md` §4 file list |

---

## 7. Absorbed items, cross-references, and notes that are not chapter claims

**Absorbed (counted once, inside the item named):**

| # | Chapter claim | Absorbed into |
|---|---|---|
| X1 | Schema-hash validation is FATAL on mismatch (§1.1, §1.2, §3.2 contract 3, §4.3, §11.1 row 2, §13) | **Chapter 6 audit B5**: `pg:350-353` passes no `survivors_file`, so the check gated at `mf:334` never runs; `validate_feature_schema_hash` is imported (`pg:68`) and never called |
| X2 | §5.1 `s['features'][f]` (a missing feature would raise) | **Chapter 6 audit B5**: missing names are filled with 0.0 (`pg:798`) |
| X3 | Three outputs (`ranked_predictions.json`, `prediction_pools.json`, `next_draw_prediction.json`) and three pools Tight 20 / Balanced 100 / Wide 300 (§1.1, §1.2, §1.3, §2.1, §2.2, §6.1, §6.3, §8.1 incl. sizes, §8.3, §13) | **PBM §8 / §16 D3** (GOVERNED as system-map Blocker 1) and **Chapter 13 audit B1** (what Chapter 13 grades is not what Step 6 writes). Commit `c6e3989` cites a Beta ruling for one canonical output. The pool tiers live in the separate backtest CLIs `build_pools.py` / `evaluate_pools.py` (PBM §8) |
| X4 | Ranked output records with `seed, rank, predicted_quality, features` (§5.2, §8.2) | **Chapter 13 audit B15 Blocker 1** (no producer for `predictions/ranked_predictions.json`). Host: `ranked_predictions_s87.json` (root, 2026-02-14) is a hand-made fixture (seed 123456, round scores), not Step 6 output (Ch14 audit row "Q1 stop 2") |
| X5 | §10.2 `handle_step6_completion` (escalate under 5 % confidence, LLM query) | **Chapter 12 audit B1**: at `a15b524`, `evaluate_results` takes the file-exists branch for every manifest (`wa@:946`), and no Step-6 handler exists (`/bin/grep -n handle_step6` → none). Repaired at `b6cb789` for failure authority (a non-zero exit, including Step 6's exit 3, now escalates as `step_failed`; `docs/CLAUDE_CODE_REPORT_WATCHER_SAFETY_REPAIRS.md`). No confidence check exists either way |
| X6 | §7.1 cross-pool combination with pool weights 0.5/0.3/0.2 | **B3** (same voting design, never built) |
| X7 | §11.4 `jq '.primary_prediction'` and `jq '.pools \| keys'` on `prediction_pools.json` | **D9** (shape) and **X3** (file) |
| X8 | §9.2 usage examples (`--pool-sizes`, `--prng-type`, `--output-dir`) | **D10** and **S1** |

**Cross-referenced, not chapter claims (not counted):**
- **Chapter 6 audit B3** — the checkpoint's NN architecture (256-128-64) differs from the sidecar's
  `hyperparameters`. Step 6 loads whatever the checkpoint holds: `nnw:406` reads `hidden_layers` from the checkpoint.
- **Chapter 13 audit B1 / A.7** — no host bridge writes `prediction_pool.json`; re-checked this pass (`find . -name
  "prediction_pool.json"` outside `backups/` → none; `crontab -l` → none; the only project unit
  `daily3scraper.service` is disabled). `wa@:2223` maps Step 6's results file to `prediction_pool.json`, so
  `_find_results(6)` never finds Step 6's real output.
- **Chapter 4 audit D9** — `extract_ml_features` accepts the forward/reverse lists and ignores them. This matters to
  Step 6 because the dual-sieve path recomputes features through it (Q2 step 2).

**Notes for the owner (not chapter claims, not counted):**
- **N1 — In the default dual-sieve path, every vote uses `default_skip`, not the survivor's own skip.**
  `_coerce_seed_list` reduces dict survivors to ints (`ss:128-137`), so `_get_survivor_skip` falls back for every
  survivor (`pg:530-553`, `:789`, `:795`). Host: 19 of 19 runs used `dual_sieve` with `default_skip 0`. The v6.0
  header calls per-survivor skip a P0 fix (`pg:25`). The fact file (`docs/TFM_PROJECT_FACTS_SKILL.md:1118-1122`,
  "regenerated with that seed's own skip") and `docs/CLAUDE_CODE_REPORT_PIPELINE_OVERVIEW.md:582` state the dict-path
  behaviour as if it were the production path. The chapter makes no per-survivor-skip claim (its §6.2 passes no
  skip), so this carries no label here. Not found in the governance trail (`/bin/grep -rn "_coerce_seed_list" docs/`
  → Chapter 4 text and audit, `IMPLEMENTATION_CHECKLIST.md`; none records the skip loss).
- **N2 — Fourteen configuration knobs are parsed and never used** in the prediction path: `ensemble_mode`,
  `ensemble_methods`, `temporal_decay`, `recency_window`, `drift_sensitivity`, `forward_weight`,
  `bidirectional_bonus`, `min_confidence`, `confidence_threshold`, `min_survivor_history`, `skip_mode_preference`,
  `feature_weights`, `use_global_features`, `predictions_dir`. Each appears only in `PredictionConfig` / `from_json`
  / `to_dict` (`pg:88-225`). `GlobalStateTracker` is imported (`pg:74`) and never used. `use_global_features` is
  False in code ("Disabled until properly wired", `pg:104`), true in `pm:163-169` and in
  `prediction_generator_config.json`. `pm` lists several of these as "optimized_by: Optuna TPE". (`predictions_dir`
  is the S1 item; it is listed here only for completeness.)
- **N3 — WATCHER passes Step 4's `reinforcement_engine_config.json` as Step 6's `--config`** (`pm:177`). `from_json`
  uses `data.get('prediction', data)` (`pg:147`). That file has no `prediction` key (keys at `888cf3e`: `model,
  training, prng, global_state, normalization, survivor_pool, output, _meta_optimizer`), so only same-named keys
  (`prng`, `output`) can reach Step 6. When the file is absent, `main` falls back to defaults (`pg:1013-1016`).
  Chapter 6 audit B7 already records Step 6 as that file's reader. This is a note, not a label.

---

## 8. Surfaces searched, and surfaces unavailable

**Governance trail searched (first):** `docs/TB_RULING_S177_KPI_GOVERNANCE.md` (Blocker 5, `:303-351`),
`docs/TB_RULING_S179_IMPLEMENTATION_AUTH.md` (§2.5, §2.6, sequence `:530-547`), `docs/TB_RULING_*` by
`/bin/grep -ln "prediction_generator\|next_draw_prediction\|tight.*balanced"`, `docs/PROPOSAL_*` (same grep; read
`PROPOSAL_S172_RANGE_MINER_v1_4_4.md` §4.4.1, §4.4.5, `PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_SEPARATION_v1_1.md`
§1–§4.1), `docs/TEAM_ALPHA_SKIP_SEMANTICS_SUBMISSION.md` (grep for Step 6), `docs/*WINDOW_ANCHOR*`, `docs/*BRIEF_I*`
(grep for Step 6 / prediction_generator → none), commit `c6e3989`.
**Chapters and audits searched:** `CHAPTER_6_AUDIT.md` (B3, B5, B7, reconciliation), `CHAPTER_13_AUDIT.md` (§1, B1,
B2, B15, B19, §8, §9, A.5, A.7, A.10), `CHAPTER_12_AUDIT.md` (B1, B3, B6, D8, row 2), `CHAPTER_4_AUDIT.md` (D9, D12,
`_coerce_seed_list`), `CHAPTER_5_AUDIT.md` (Step 6 mentions), `CHAPTER_9_AUDIT.md` (row 22), `CHAPTER_14_AUDIT.md`
(Step 6 mentions), `CHAPTER_1_AUDIT_v1.md` (C-5), `CHAPTER_3_ALIGNMENT_AUDIT.md` (grep), `PIPELINE_BEHAVIOUR_MODEL.md`
(§1.1–§1.3, §3.6, §8, D3, I-2, I-3), `TODO_CONSOLIDATED.md` (grep: no Step 6 item), `CHAPTER_2_BIDIRECTIONAL_SIEVE.md`
(shape only; not edited).
**Code read:** `prediction_generator.py` (all 1,095 lines), `models/model_factory.py` (`:1-130`, `:260-377`),
`models/feature_schema.py` (`:45-210`), `survivor_scorer.py` (`:100-200`, `:350-375`), `models/wrappers/` (listing;
`neural_net_wrapper.py` `load`), `integration/metadata_writer.py` (`:40-165`), `agent_manifests/prediction.json` (all),
`agents/watcher_agent.py` at `a15b524` (`:170-185`, `:380-420`, `:474`, `:760-835`, `:946`, `:1380-1400`, `:1700-1900`,
`:2078-2262`), `window_optimizer_integration_final.py` (`:2900-2990`), `utils/canonical_arrays.py` (grep).
**Gitignored / host (read-only):** `agent_manifests/definitions.json` (`:60-110`); `predictions/` and
`predictions/history/` (19 files: keys and metadata only); `models/reinforcement/` (listing, sidecar keys);
`forward_survivors.json`, `reverse_survivors.json`, `prediction_generator_config.json` (contents);
`ranked_predictions_s87.json`, `prediction.json` (root; heads); `logs/` (grep for Step 6 EXEC CMD; `s120_step4_6.log`);
`git ls-files -o -i --exclude-standard` (predict*); `crontab -l` (none); `systemctl list-unit-files`
(`daily3scraper.service` disabled); `/etc/systemd/system`; `~/.config/systemd/user`.
**Git history:** `git log` of `prediction_generator.py` (12 commits), `-S"build_prediction_pools"` (all refs),
`-S"objects not retained"`, `-S"S166-ACCUM"`, `-S"_param_to_cli"`, `-S"tolerates summary stub"`;
`git show 888cf3e:reinforcement_engine_config.json` (keys).

**Not available, and where it matters**
- **Runtime behaviour.** Nothing was executed. B1's `ValueError` is traced by reading `ss:128-137` against the
  host stub's exact content, not by running Step 6.
- **The text of the 2026-01-28 Beta ruling** (S1) — cited in code and in a commit; no file found.
- **`.127`, the rigs, root and other users' crontabs** — UNAVAILABLE, not reported as clean. Step 6 is not
  distributed, so the rigs bear on nothing here.
- **Pre-repository archives and the private remote** — not read.

---

## 9. Accurate claims, unverified claims, and count reconciliation

**Accurate (18)**
1. Step 6 in the executable scheme (title, §1.1) — `wa@:394`; `pm:4`.
2. Purpose: generate final predictions with the trained model (header) — `pg:1-15`, `pg:828`.
3. Loads the Step 5 model through sidecar metadata (§1.1, §2.1 inputs `best_model.*` + `meta.json`) — `pg:290-297`,
   `pg:350-353`; `mf:262-376`.
4. Model type from the sidecar only, never from the extension (§1.2, §3.1, §13) — `mf:308`, `mf:374`, `mf:98`.
5. Missing sidecar → `FileNotFoundError` (§3.2 contract 1) — `pg:292-294`; `mf:291-296`.
6. Missing `model_type` → `ValueError` (§3.2 contract 2; §11.1 "Model type missing → re-run Step 5") — `mf:302-306`.
7. Survivors ranked by predicted quality, best first (§1.1, §5.1 sort) — `pg:828`, `pg:836`.
8. Per-prediction confidence scores (§1.1, §1.2) — `pg:686-713`.
9. Why order matters: features are placed by the sidecar's names (§4.1) — `pg:366`, `pg:797-798`.
10. Hash = SHA-256 of the lexicographically sorted names, 16 hex (§4.2) — `fs:99`, `fs:185-188`; `fs:207`.
11. Values regenerated through `SurvivorScorer(prng_type, mod)` (§6.2) — `pg:263-266`, `pg:846`.
12. Defaults `java_lcg`, mod 1000 (§6.2 signature, §9.1 defaults) — `pg:107-108`, `pg:164-165`.
13. Output `predictions/next_draw_prediction.json` (§2.1 third output, §11.4 path) — `pg:905-907`; `pm:12`.
14. Lineage: `parent_run_id` from the Step 5 sidecar, `pipeline_step 6` (§10.3) — `pg:302-303`, `pg:936-937`.
15. Terminal step, no follow-up agent (§10.1, §10.3, §13) — `pm:39`; `pg:938`; `mw:156-158`; `wa@:1390-1393`.
16. `--survivors-forward` and `--lottery-history` are required to predict (§9.1) — `pg:1009-1010`.
17. Upstream chain: Step 3 → `survivors_with_scores.json`, Step 5 → `best_model.*` + `best_model.meta.json` (§1.3)
    — PBM §1.3 rows 3 and 5; `pm:7-8`.
18. The sidecar carries `model_type`, `checkpoint_path` and `feature_schema{source_file, feature_count,
    feature_names, ordering, feature_schema_hash}` (§3.3 structure) — host sidecar; `fs:97-100`.

**UNVERIFIED (1)**
- **U1** Expected hit rates and success metrics: Tight "highest lift", Hit@100 > 70 %, Hit@300 > 90 %, lift 10–20×
  (§6.1, §8.3, §13). No artifact measures them. No Step 6 output has ever been graded on the host (Chapter 13
  audit A.7: 233 diagnostics, all synthetic, all with an empty pool). Beta records that independent Hit@100/300
  cannot be derived from a 20-entry pool (`TB_RULING_S177_KPI_GOVERNANCE.md:325`). Would verify: a graded
  walk-forward with independent Hit@K as S179 §2.6 defines it. No conclusion about the figures is drawn here.

**Count reconciliation**

| Category | Count | Items |
|---|---|---|
| Accurate | 18 | §9 list 1–18 |
| BROKEN | 3 | B1, B2, B3 (B3 governed by S179 §2.5) |
| SWITCHED-OFF-BY-RULING | 1 | S1 |
| DESCRIPTION-ONLY | 16 | D1–D16 |
| UNVERIFIED | 1 | U1 |
| Absorbed into an item | 8 | X1–X8 (§7 table) |
| **Total claims checked** | **47** | 18 + 3 + 1 + 16 + 1 + 8 = 47 |

N1–N3 and the cross-references in §7 are not chapter claims and are not counted. No UNVERIFIED item was resolved
during the pass without a final placement: U1 stays UNVERIFIED.

**Files opened (execution proof):** listed in §8. This pass edited nothing but this file and
`docs/CHAPTER_7_PREDICTION_GENERATOR.md`.
