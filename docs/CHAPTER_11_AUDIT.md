# Chapter 11 audit — feature importance and visualization

Repo: `/home/michael/distributed_prng_analysis` on VM 101 (`192.168.3.177`), user `michael`, read-only.
Code base: `a15b5247336255801de8cf67d52653701b1e8c80` (`a15b524`). Working-tree HEAD during this pass was `b6cb789`,
which changed exactly three non-doc files (`agents/watcher_agent.py`, `agents/watcher_dispatch.py`,
`tests/test_watcher_safety_repairs.py`; `git diff --name-only a15b524 HEAD`). The one `agents/watcher_agent.py` anchor
used here (`STEP_SCRIPTS`) was taken from `git show a15b524:agents/watcher_agent.py`. Every other code file cited is
identical at `a15b524` and HEAD (`git diff --quiet a15b524 -- <file>` returned 0 for `feature_importance.py`,
`feature_visualizer.py`, `feature_drift_tracker.py`, `feature_importance_ai_interpreter.py`,
`feature_importance_interpreter.py`, `per_survivor_attribution.py`, `web_dashboard.py`, `modules/visualization_manager.py`,
`modules/web_visualizer.py`, `meta_prediction_optimizer.py`, `meta_prediction_optimizer_anti_overfit.py`,
`survivor_scorer.py`, `full_scoring_worker.py`, `chapter_13_diagnostics.py`, `utils/canonical_records.py`). None of the
three b6cb789 repairs (Ch12 B1, B3, B6) touches anything this chapter describes.
Audited: `docs/CHAPTER_11_FEATURE_IMPORTANCE_VISUALIZATION.md` (1099 lines at `a15b524`) against the code.
Authority: `docs/CLAUDE_CODE_INSTRUCTIONS_CHAPTER_CORRECTIONS_20261111.md` (REV1), §0, §1, §3, §5.
Method: governance trail first, then chapters, then code, then host state. Every anchor below was opened in this
session. Nothing was executed except read-only `python3 -I` one-liners over host JSON artifacts (keys and name sets
only for `models/reinforcement/best_model.meta.json` and `diagnostics_outputs/`) and an `ast` parse of the two name
lists in `feature_importance.py`. No training, dashboard, web server, LLM query or pipeline step was started. After the
edits, the Chapter 2 clean control `python3 tests/test_chapter2_content_gate.py` reported "12/12 checks green",
`SENTINEL : PASS`; `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` is untouched (`git diff --quiet`).
Numbering: Chapter 11 is a **component/analysis chapter**; it documents no executable step. Bare "Step N" below means
the **executable** scheme (`docs/PIPELINE_BEHAVIOUR_MODEL.md` §1.1; `git show a15b524:agents/watcher_agent.py`
`:387-395` `STEP_SCRIPTS`, Step 5 = `meta_prediction_optimizer_anti_overfit.py` at `:393`).
Framing held throughout: TFM is functional mimicry of surface output. An importance value is a statement about the
mimicry model, not about the machine; no finding here assumes a "true seed".

---

## 1. Verdict

Null results first:

- **Nothing in the pipeline produces global feature importance at `a15b524`.** The Step 5 writer of
  `feature_importance_step5.json` was removed at `e0b55a1` (2025-12-20) and the import at `249052b` (2025-12-22);
  the approved Addendum H still says "IMPLEMENTED" (B1). Denominator: every tracked `*.py`/`*.sh`/`*.json`
  (`/bin/grep -rn`, so gitignored files included), git history (`git log --all -S`), host. The only code caller of
  `get_feature_importance()` is `meta_prediction_optimizer.py:195`, which is not in `STEP_SCRIPTS` and has no caller.
- **Every consumer reads files with no producer.** `feature_visualizer.py`, the dashboard `/features` route, and both
  interpreters read `feature_importance_step4.json` / `_step5.json` / `feature_drift_step4_to_step5.json`. The host
  copies are dated 2025-12-10, carry 24 names of which **4 of 24** exist in the live 91-feature vector, and have no
  writer in code or history.
- **The global `SHAP` and integrated-gradient methods the chapter documents were never in code** (B2, B3):
  `git log --all -S` for `compute_shap_values` and `compute_integrated_gradients` hits only documentation commits.

Counts (33 claims; full reconciliation in §10):

- Accurate: **10**
- BROKEN: **4**
- SWITCHED-OFF-BY-RULING: **0** (the governed holds that bear on this chapter are already counted in the Chapter 6,
  13 and 14 audits and are absorbed here)
- DESCRIPTION-ONLY: **12**
- UNVERIFIED: **1**
- Absorbed into an existing item, not counted separately: **6**

What matters most:

1. The extraction module is sound on reading: native, permutation and gradient importance are implemented and
   normalised (`feature_importance.py:126-499`). What is missing is a caller in the pipeline (B1).
2. Its category accounting cannot place the live feature names. With a real 89-name importance dict,
   `global_state_features` sums to 0 and 45 of 89 names fall into `other_features` (B4).
3. The chapter's chart catalogue, CLIs, output files, LLM and function names are a design that was never built or
   was later changed (D2, D4, D7–D10). The real visualizer is 13 Plotly functions over importance dicts only.
4. The feature contract is **91 extracted / 89 trained**, re-derived from code (§2 N3), not 62.

---

## 2. Numbering, canonical version, manifests, fact-file statements

| # | Question | Result | Evidence |
|---|---|---|---|
| N1 | Numbering scheme | Component chapter; executable scheme for bare "Step N". The file names `feature_importance_step4.json` / `_step5.json` use the **Addendum H era** numbering, in which "Step 4" was `meta_prediction_optimizer.py`; at `a15b524` executable Step 4 is `adaptive_meta_optimizer.py` | `git show a15b524:agents/watcher_agent.py:392-393`; `docs/PROPOSAL_Addendum_H_Feature_Importance_Integration.md:48-58` |
| N2 | Canonical version | Only one Chapter 11 file ever committed. History: added `53a3829` (2026-01-08); LLM section updated `49e8ad4` (2026-01-20, per `docs/PATCH_Chapter11_LLM_Update_v2.md`); **that update reverted by "Doc sync 20260209_1704" `323b6fa`** (same 12 lines, opposite direction). The file at `a15b524` is the pre-update text | `git log --all -- docs/CHAPTER_11_*`; `git show 49e8ad4`, `git show 323b6fa` on the chapter |
| N3 | Feature count, re-derived from code | **91 extracted**: 27 match/statistics keys (`survivor_scorer.py:702-731`) + 9 residue (3 mods × 3, `:74`, `:625-667`) + 23 battery (`:756`) + 18 merged/defaulted NPZ and placeholder fields (merge `:774-779`, defaults `:783-791`) + 14 `global_*` (`full_scoring_worker.py:403-405`, `:458-460`). **89 trained** = 91 − `score`, `confidence` (`meta_prediction_optimizer_anti_overfit.py:1479-1483`, `:546-597`; `holdout_hits`/`holdout_quality` are listed but are top-level keys, not in `features`). Host: first record of `survivors_with_scores.json` (2026-03-11) has 91 feature keys; sidecar `feature_schema.feature_names` has 89 entries (length only) | as cited; agrees with `docs/S172_ATTRIBUTION_AND_FEATURE_TRACE_REPORT.md:225-272` |
| N4 | Manifests | No manifest or `definitions.json` entry for feature importance. `agent_manifests/definitions.json` (gitignored, 220 lines, 2026-01-01) carries an example `"feature_count": 62` (`:178`) | `/bin/grep -n -i` over `agent_manifests/*.json` |
| N5 | Fact-file statement | `docs/TFM_PROJECT_FACTS_SKILL.md:2860` "feature_importance.py's 60-name list being stale by 31" — **confirmed** (31 with the `global_` prefix applied; 45 without, B4) | `ast` parse of `feature_importance.py:95-119` against the live name set |

---

## 3. What actually works end to end today

1. `get_feature_importance(model, X, y, feature_names, method='auto', ...)` dispatches to native (tree), permutation
   (any model; custom GPU loop for PyTorch) or gradient (PyTorch) importance and returns a sorted, normalised dict
   (`feature_importance.py:126-213`, `:220-295`, `:302-499`). `FeatureImportanceExtractor.extract` wraps it with
   metadata (`:506-599`). Works on reading; **no pipeline caller** (B1).
2. `feature_drift_tracker.py` computes mean |Δ| drift between two importance dicts with threshold 0.15 and keeps a
   history file (`:228-460`, `:513-557`). No pipeline caller (searched).
3. `feature_visualizer.py` renders 13 Plotly charts from importance files into one `feature_report.html`
   (`:1265-1423`) and exposes the same functions to `web_dashboard.py` `/features` (`:38-57`, `:2033-2088`).
   It reads the B1 files only.
4. `feature_importance_ai_interpreter.py` builds a prompt from `feature_importance_step5.json` and tries DeepSeek-R1-14B
   on `localhost:8080`, then the Claude CLI, then a template; it prints to stdout (`:172-205`, `:208-297`).
   `feature_importance_interpreter.py` is a template-only variant (`:1-60`).
5. Per-survivor attribution (`per_survivor_attribution.py`) is a separate module (Chapter 14 Phase 2) and does not
   execute (Ch13 B15; Ch14 S2).

Host (read-only, 2026-11-11): no dashboard on `:5002`, nothing listening on `:8080`/`:8081` (`ss -ltn`); no cron
(`crontab -l`); no TFM/dashboard unit (`systemctl list-units --all`, only `nvidia-gpu-policy.service`, failed).

---

## 4. BROKEN

"Governed" = already recorded in the governance trail. None of B1–B4 is held by a ruling.

**B1 — No pipeline step produces global feature importance; every consumer reads files nothing writes.**
*(Not governed. The approved design says the opposite: `docs/PROPOSAL_Addendum_H_Feature_Importance_Integration.md:6`,
`:60-70`, "Status: IMPLEMENTED", Step 5 → `feature_importance_step5.json`.)*
- Chapter: importance analysis runs after Step 5 training (§1.1 `:35`, §2.1 `:65-86`, §11.1 `:841-861`).
- Code: `meta_prediction_optimizer_anti_overfit.py` contains no reference to `feature_importance`
  (`/bin/grep -n`). The writer was removed at `e0b55a1` ("v1.7.0: Critical label leakage fix") and the import at
  `249052b` ("Session 11: Subprocess isolation"); neither commit message mentions it. The only remaining code caller,
  `meta_prediction_optimizer.py:41`, `:195`, is not in `STEP_SCRIPTS` (`git show a15b524:agents/watcher_agent.py:387-395`)
  and is invoked by nothing but a test docstring (`test_feature_importance_zeus.py:3`).
- Consumers of the missing files: `feature_visualizer.py:69-72`, `:158-159`, `:418`, `:864-865`, `:1208-1209`,
  `:1269-1271`; `web_dashboard.py:2043-2045`; `feature_importance_ai_interpreter.py:209-210`;
  `feature_importance_interpreter.py:11-12`.
- Surfaces searched for a writer: tracked and gitignored `*.py`/`*.sh`/`*.json` (`/bin/grep -rn
  'feature_importance_step[45]|feature_drift_step4_to_step5'` → readers only); git history (`git log --all -S
  'feature_importance_step5'` → `49e8ad4`, `249052b`, `e0b55a1`, `1ddeb75`, `0e82155`, `3d896a5`, `392eadb`,
  `4ada59f`, `6175e09`, `8fec6d5`); host (the three files exist, 2025-12-10; their key shape `pipeline_step`,
  `model_type`, `extraction_method` is not the removed writer's shape `model_version`, `top_10`, `summary`
  (`git show e0b55a1`); 24 names, 4 of 24 in the live vector; `git log --all -S 'lane_agreement_16'` → nothing).
- Stops at: no caller. Repair would have to: name one pipeline producer that writes the file shape the readers expect,
  keyed by the sidecar's 89 names, or state that the visualizer and interpreters are standalone tools over a
  hand-supplied file.

**B2 — Global SHAP importance is not implemented.** *(Not governed; never in code.)*
- Chapter: "3 Importance Methods: Permutation, Gradient, SHAP" (§1.3 `:53`), §4.1 `:205`, §7.2–7.3 `:431-524`,
  `--method shap|all` (§12.1 `:899`), `shap_*.png` (§13.2 `:994-996`), `compute_shap_values()` / `generate_shap_plots()`
  (§15.1 `:1044-1045`), SHAP waterfall/force charts (§9.1 `:591-592`).
- Code: `_resolve_method` accepts `auto`, `native`, `permutation`, `gradient` and raises on anything else
  (`feature_importance.py:261-295`). No tracked `*.py` imports `shap` (`/bin/grep -rln '^\s*(import shap|from shap)'` → 0).
  `git log --all -S compute_shap_values` → `53a3829` only (the chapter). Per-survivor SHAP-type values exist elsewhere
  (CatBoost `ShapValues`, `per_survivor_attribution.py:259`; XGBoost/LightGBM contributions `:163`, `:206`) — that is
  X3, not this claim.
- Repair would have to: implement a global SHAP path, or state that global importance is native/permutation/gradient only.

**B3 — Integrated gradients are not implemented.** *(Not governed; never in code.)*
- Chapter: §6.3 `:382-418`, §15.1 `:1043`.
- Code: `/bin/grep -rn integrated_gradients` over tracked `*.py` → 0; `git log --all -S compute_integrated_gradients` →
  `53a3829` only.
- Repair would have to: implement it, or state that gradient importance is plain saliency (`feature_importance.py:465-499`).

**B4 — Category accounting cannot place the live feature names.** *(Recorded as "STALE" in
`docs/S172_ATTRIBUTION_AND_FEATURE_TRACE_REPORT.md:638` and the fact file `:2860`; no ruling holds it. The
zero-global-weight consequence is added here.)*
- Chapter: importance analysed by category (§3.2 `:167-177`; §10.1 question 3 `:789`; §10.2 "Category Analysis" `:821-827`).
- Code: `_compute_category_importance` and `get_importance_summary_for_agent` sum over `STATISTICAL_FEATURES`
  (46 names) and `GLOBAL_STATE_FEATURES` (14 names) (`feature_importance.py:95-119`, `:601-633`, `:679-705`); the same
  lists are duplicated in `feature_drift_tracker.py:197-221` and drive `statistical_drift`/`global_drift` (`:429-436`).
  The global list is **unprefixed** (`regime_age`); the live names are `global_regime_age` etc.
  (`full_scoring_worker.py:403-405`). Parsed against the live 89 trained names: 46 of 46 statistical names present,
  **0 of 14** global names present, 45 of 89 trained names in neither list. With the `global_` prefix applied the 14
  globals match and 31 remain unlisted: 23 `batt_*`, 4 digit features (`hundreds_/tens_/ones_digit_agreement`,
  `expected_digit_match_count`), `bidirectional_count`, `bidirectional_selectivity`, `skip_min`, `skip_max`.
  So for any real model `global_state_features` = 0, `global_drift` = 0, and the dashboard's category pie
  (`feature_visualizer.py:403-470`, via `get_importance_summary_for_agent`) shows 0 % global.
- Repair would have to: derive the category lists from the sidecar's `feature_schema.feature_names` (or the Step 3
  producer), or state that the category split is a Dec-2025 60-name view.

---

## 5. SWITCHED-OFF-BY-RULING

None counted here. Three governed holds bear on what this chapter describes; each is already counted in another
audit and is absorbed (§9, X2, X3, X5). For the chapter's "Known gaps" list they are restated with their rulings:

- **G5, feature importance in the sidecar** — TB approved all six S111 guardrails, owner held the patch
  (`docs/SESSION_CHANGELOG_20260225_S111.md:80-84`; `docs/S111_TEAM_BETA_BRIEFING.md:127-128`; Ch14 audit
  Addendum 2). Host: `best_model.meta.json` has no `feature_importance` key (keys read). Consumer that would light up:
  `chapter_13_diagnostics.py:449` (`compute_feature_diagnostics` reads `model_meta["feature_importance"]`, today `{}`,
  so turnover and dominant-shift are always 0). *If lifted:* every trained model would carry its own importance, and
  Chapter 13's feature-drift diagnostic would have an input.
- **Real per-survivor attribution run** — blocked (Ch14 S2; system map `:272`, `:275`); attribution-driven autonomy
  not authorized (Ch14 S4, `:277`). *If lifted:* per-seed attributions; needs P0-C first (Ch13 B15).
- **`y = holdout_quality` replacing `holdout_hits`** (Ch6 S1). *If lifted:* Step 5 would train on exact holdout hits again.
- Related deferral, not a prohibition: web dashboard `/training` route and TensorBoard (Ch14 S9). The `/features`
  route this chapter's visualizer feeds does exist (`web_dashboard.py:2033`).

---

## 6. DESCRIPTION-ONLY

| # | Chapter | Code at `a15b524` |
|---|---|---|
| D1 | "Files: `feature_importance.py`, `feature_visualizer.py`", "Lines: ~600 combined" (`:5-6`); §1.2 "The Two Files" | 724 + 1422 = 2146 lines (`wc -l`). The subsystem has six files: also `feature_drift_tracker.py` (663), `feature_importance_ai_interpreter.py` (332), `feature_importance_interpreter.py` (140), and the `/features` route in `web_dashboard.py` (`:2033-2088`). Drift tracking (`feature_drift_tracker.py:228-460`; `compare_importance`, `feature_importance.py:640-676`) is not mentioned at all |
| D2 | "LLM-powered analysis via Qwen2.5-Math", `localhost:8081/v1/completions`, `generate_ai_interpretation()` (§1.3 `:56`, §10.1 `:759-804`, §10.2 `:809`, §16 `:1084`) | DeepSeek-R1-14B at `localhost:8080/completion` → Claude CLI → template (`feature_importance_ai_interpreter.py:3-13`, `:27`, `:36-48`, `:172-205`); `llm_services/llm_server_config.json` `primary.port` 8080, `backup` Claude CLI. Entry point `interpret_with_ai(step5_path, drift_path)` (`:208`), output to stdout only (no `ai_interpretation.txt`; the only `open()` calls read, `:219`, `:223`). The chapter was updated to this at `49e8ad4` and reverted by `323b6fa` (N2) |
| D3 | "4 Model Types: Neural Net, XGBoost, LightGBM, CatBoost" (§1.3 `:54`); methods table omits `native` | Detection is by interface: `pytorch`, `xgboost`, `sklearn_tree` (anything with `feature_importances_`), `sklearn_other` (`_detect_model_type`, `:220-246`). `auto` → `native` for trees (XGBoost gain, `:331-359`; `feature_importances_`, `:302-328`), `permutation` for PyTorch (`:272-279`). Whether Step 5's actual saved objects pass through it is U1 |
| D4 | Export "PNG, HTML (interactive), JSON, CSV" (§1.3 `:57`); `visualization_outputs/` with 16 files (§2.1 `:81-85`, §13.2 `:982-1000`); "`pip install plotly kaleido`" (§14.1 `:1014`) | Charts are HTML fragments (`fig.to_html(full_html=False, include_plotlyjs="cdn")`, e.g. `feature_visualizer.py:150`) assembled into one `feature_report.html` in the working directory (`--output` default, `:1272`, written `:1414-1416`). No PNG (`write_image`/`savefig`: 0 hits), no CSV, no `visualization_outputs` anywhere in tracked code. JSON only via `FeatureImportanceResult.save` (`feature_importance.py:78-81`). `kaleido` is not used |
| D5 | Checkpoint "`best_model.pth` (or .xgb, .lgb, .cbm)" (§2.1 `:68`) | `.pth`, `.json`, `.txt`, `.cbm` (`models/model_factory.py:29-32`; host `models/reinforcement/` listing) |
| D6 | "62 Total Features" = 48 per-seed + 14 global, with names (§3.1 `:106-165`; also `:192`, `:313`, `:815`, `:961`) | 91 extracted / 89 trained (N3). Globals are `global_*`; `marker_567_variance`, `regime_stability`, `entropy` do not exist (live list). The "62" matches the post-`e0b55a1` count ("64 → 62 total") and Addendum H's 60-name list. Cross-ref Ch5 D2, `docs/PROJECT_FILE_CATALOG.md:933` |
| D7 | `compute_permutation_importance()` with `scoring='neg_mean_absolute_error'` (§5.2 `:246-287`); `compute_gradient_importance()` (§6.2 `:339-380`); §15.1 function table | Public entry is `get_feature_importance()` (`:126`); private `_extract_permutation_importance` uses `neg_mean_squared_error`, `random_state=42` (`:387-392`); the PyTorch path is a custom GPU loop on MSE with unseeded `torch.randperm` (`:405-462`); `_extract_gradient_importance` (`:465-499`). `compare_model_importance()` does not exist; `compare_importance()` is drift between two dicts (`:640-676`) |
| D8 | `class FeatureVisualizer` with 13 methods and the §9.1 catalogue (bar, radar, correlation matrix, feature distribution, pairwise scatter, category heatmap, cumulative/Pareto, violin, boxplot, SHAP waterfall, SHAP force, dendrogram, 2×2 dashboard) (§8 `:528-572`, §9 `:576-751`, §15.2 `:1048-1064`) | No class. 13 module-level functions over **importance dicts only** — no feature matrix `X` is ever read: `importance_bar`, `drift_comparison`, `drift_gauge`, `top_movers`, `category_pie`, `scatter` (Step 4 vs Step 5), `treemap`, `radar`, `dual_radar`, `heatmap` (importance over runs, `:846-937`), `histogram` (of importance values), `waterfall` (drift change), `sunburst` (`feature_visualizer.py:1283-1297`). All Plotly. Correlation matrix, value distributions, pairwise scatter, Pareto, violin, boxplot and dendrogram are not produced. The SHAP charts belong to B2 |
| D9 | CLIs: `feature_importance.py --model-path … --method …` (§11.1, §12.1 `:893-905`, §12.3); `feature_visualizer.py --importance-json --data-file --format --dpi --output-dir` (§12.2 `:910-921`); Step 5 `--n-trials 50` (§11.1 `:848`) | `feature_importance.py` has no CLI — `__main__` is a self-test (`:712-724`). `feature_visualizer.py` flags: `--step4 --step5 --drift --output --top-n --charts` (`:1268-1274`). Step 5 has `--trials` (`meta_prediction_optimizer_anti_overfit.py:3045`) and requires `--lottery-data` (`:3043`), so the §11.1 command fails at argparse |
| D10 | `importance.json` shape `{method, model_type, n_features, timestamp, importance, metadata}` (§13.1 `:957-977`) | `FeatureImportanceResult` fields `computation_method, model_version, timestamp, total_features, importance_by_feature, importance_by_category, top_10_features, bottom_10_features, model_type_detected` (`feature_importance.py:62-73`). Readers accept `feature_importance`, `importance_by_feature` or a bare dict (`feature_visualizer.py:85-91`) |
| D11 | "Chapter 14: Feature Importance & Visualization" (§16 `:1070`, footer `:1099`); "Next Chapter: Chapter 16: Prediction Generator" (`:1091`) | This is Chapter 11. The Prediction Generator is Chapter 7 (`docs/CHAPTER_7_PREDICTION_GENERATOR.md`); Chapter 12 (WATCHER) follows this one |
| D12 | Prompt "a PRNG seed prediction model" (§10.1 `:776`); "What drives predictions … WHY the model works" (§1.1, §2.2) | Importance is a statement about the **mimicry model** (which inputs the fitted model uses to rank survivors), not about the generator or machine. The code's own prompt says "how the PRNG can be predicted" (`feature_importance_ai_interpreter.py:260`) — code text, recorded in §11, not changed |

Note N1 (not counted; outside the chapter's text). The drift tracker reports `status` `'ALERT'`/`'stable'`
(`feature_drift_tracker.py:602`); the visualizer colours `'drift_detected'` (`feature_visualizer.py:285`, `:1228`), a
value the tracker never emits. Only the colour is affected.

---

## 7. Host questions (instruction §5) and Chapter 6 / 13 / 14 cross-references

- **`agent_manifests/definitions.json`** (gitignored; read on host): no feature-importance, visualizer or drift entry;
  example `feature_count: 62` at `:178` (N4).
- **`models/reinforcement/best_model.meta.json`** (keys only): top level `schema_version, model_type, checkpoint_path,
  checkpoint_format, feature_schema{source_file, feature_count, feature_names[89], ordering, feature_schema_hash,
  excluded_features[4]}, signal_quality, data_context, training_metrics, hyperparameters, optuna, hardware,
  training_info, agent_metadata, provenance`. **No `feature_importance` key** (G5, held) and no top-level
  `feature_names` (Ch13 B15 blocker 2).
- **`diagnostics_outputs/`** (keys only): 223 JSON files; **4 of 223** carry a `feature_importance` key, all
  `history/catboost_20260209_*.json` — consistent with Ch14 B4/B5 (f{i}-keyed, CatBoost only). No `tier_comparison.json`.
- **Ch14 B4 / B5 / B18:** the training-diagnostics importance route is the only other importance producer in the tree
  (`train_single_trial.py:144-148`, `f{i}` keys; `training_diagnostics.py:651-680`); absorbed as X1.
- **Ch14 Q5:** 13 trial-level NPZ fields are in `X`; their importance is trial-level, not per-seed. Bears on every
  example ranking in the chapter (X6). Also the 9 constant columns (Ch14 audit, constant-feature table): permutation
  importance of a constant column is 0 by construction, so `skip_entropy` (a 0.0 placeholder, `survivor_scorer.py:783-791`)
  cannot rank second as in §5.3.
- **Ch13 B15 / Ch14 S2, S4:** per-survivor attribution (X3).
- **Ch14 S9:** dashboard `/training`/TensorBoard deferred; `/features` exists.
- **Ch14 Addendum 2 (G5):** X2.
- Scheduler surfaces: `crontab -l` → none; `systemctl list-units --all` → no unit for the dashboard, WATCHER or LLM;
  `ss -ltn` → nothing on 5002/8080/8081.

---

## 8. Surfaces searched, and surfaces unavailable

**Searched and read**
- Governance: `/bin/grep -l -w -i -E 'SHAP|feature_importance|feature importance|feature_visualizer|integrated gradients'`
  over `docs/TB_RULING_*`, `docs/PROPOSAL_*`, `docs/TEAM_ALPHA_*`, `docs/proposals/*` → 10 hits, all PROPOSAL files;
  read `PROPOSAL_Addendum_H_Feature_Importance_Integration.md` (all); grep-read the feature-importance sections of
  `PROPOSAL_Unified_Agent_Context_Framework_v3_2_3.md` / `v3_2_4.md` (`:1582-1728`, `:2035-2058`). No `TB_RULING_*`
  names feature importance (the unbounded `shap` grep matched "shape" in 15 rulings; the word-bounded grep matched none).
  `docs/S111_TEAM_BETA_BRIEFING.md` (§3, §5 G1–G6), `docs/SESSION_CHANGELOG_20260225_S111.md` (`:30-84`);
  `docs/S172_ATTRIBUTION_AND_FEATURE_TRACE_REPORT.md` (§1.3–1.5, §2 rows); `docs/PATCH_Chapter11_LLM_Update_v2.md`
  (outline, §1–2), `docs/apply_chapter11_patch.sh` (head); `/bin/grep` for removal records
  (`importance … removed|disabled|dropped|deferred|stub`) over `docs/`, root `*.md`/`*.txt` → 0.
- Chapters and audits: this chapter (every line); `CHAPTER_2_BIDIRECTIONAL_SIEVE.md` (model, not edited);
  `CHAPTER_13_AUDIT.md` (layout, B15, §9, A.10); `CHAPTER_14_AUDIT.md` (Q3–Q5, B4, B5, S1–S10, Addendum 2);
  `CHAPTER_6_AUDIT.md` (S1, D5); `CHAPTER_5_AUDIT.md` and `CHAPTER_5_ML_ARCHITECTURE_OPTIMIZER_v2.md` (form);
  `/bin/grep` for Chapter 11 mentions in all `CHAPTER_*_AUDIT.md`, `CHAPTER_14_TRAINING_DIAGNOSTICS.md` (`:52-53`,
  `:164`, `:610`), `CHAPTER_3_SCORER_META_OPTIMIZER.md:1325`; `PIPELINE_BEHAVIOUR_MODEL.md` §1.1; `TODO_CONSOLIDATED.md`
  (no Chapter 11 row; `:227` notes `web_dashboard.py:49` is the feature-importance heatmap); `TFM_PROJECT_FACTS_SKILL.md:2860`;
  `PROJECT_FILE_CATALOG.md:933`.
- Code, read in full: `feature_importance.py`, `feature_importance_ai_interpreter.py`. In the relevant parts:
  `feature_visualizer.py` (`:1-160`, every `def`, `:157-200`, `:403-430`, `:586-620`, `:846-880`, `:1089-1130`,
  `:1195-1423`), `feature_drift_tracker.py` (`:1-36`, `:180-230`, `:375-460`, `:513-560`, `:600-663`),
  `feature_importance_interpreter.py` (`:1-60`), `web_dashboard.py` (`:30-70`, routes, `:2033-2090`, `:2240-2259`),
  `per_survivor_attribution.py` (`:1-48`, `def` list), `meta_prediction_optimizer.py` (`:1-30`, `:180-240`),
  `meta_prediction_optimizer_anti_overfit.py` (`:546-700`, `:1476-1486`, argparse), `survivor_scorer.py`
  (`:74`, `:620-668`, `:698-795`), `full_scoring_worker.py` (`:340-352`, `:388-405`, `:458-460`), `chapter_13_diagnostics.py`
  (`:430-480`), `train_single_trial.py` (`:135-150`, `:826-836`), `utils/canonical_records.py` (`:210-258`),
  `models/model_factory.py` (`:29-32`), `llm_services/llm_server_config.json` (keys), `agent_manifests/*.json` (grep).
- Git history: `git log --all` of the chapter and of the three core modules; `git show` of `49e8ad4`, `323b6fa` (chapter),
  `4ada59f` (stat), `249052b` and `e0b55a1` (Step 5 removals); `git log --all -S` for `feature_importance_step5`,
  `compute_shap_values`, `class FeatureVisualizer`, `compute_integrated_gradients`, `generate_ai_interpretation`,
  `"extraction_method"`, `lane_agreement_16`, and `from feature_importance import` on both optimizers.
- Gitignored / host: `git status --ignored`, `git ls-files -o -i --exclude-standard` (feature files);
  `feature_importance_step4.json`, `feature_importance_step5.json`, `feature_drift_step4_to_step5.json` (keys and
  name sets); `survivors_with_scores.json` (first record's key set, streamed); `models/reinforcement/` (listing,
  sidecar keys); `diagnostics_outputs/` (listing, key search); `agent_manifests/definitions.json`; `~/dashboard_work`
  (`/bin/grep -rl`: one design note citing `/features` as a precedent, `DESIGN_DASHBOARD_MINER_v1.md:57-119`);
  `crontab -l`; `systemctl list-units --all`; `ss -ltn`.

**Not available, and where it matters**
- **Runtime behaviour** of `get_feature_importance()` on the objects Step 5 saves (raw `lgb.train`/`xgb.train`
  Boosters, the NN dict checkpoint, CatBoost) — not executed by scope (U1).
- **Whoever wrote the 2025-12-10 host files** — no writer in code or history; origin UNAVAILABLE.
- **Running dashboard / LLM server** — not started by scope; both down at the time of the check.
- **ser8 archives, private remote, rigs** — not consulted (no SSH by brief).
- **Not read:** `modules/visualization_manager.py` and `modules/web_visualizer.py` beyond confirming they do not import
  any Chapter 11 module (`/bin/grep -rn` for the module names returned no hit in them); the `apply_s*` patch corpus.

---

## 9. Accurate claims and unverified claims (for the count)

**Accurate (10)**
1. `feature_importance.py` computes importance; `feature_visualizer.py` generates charts (§1.2 `:44-47`) — `feature_importance.py:126`; `feature_visualizer.py:42`.
2. 13 chart types (count) (§1.2, §1.3 `:55`, §8.1) — `feature_visualizer.py:1283-1297`. (Set differs: D8.)
3. Permutation importance: shuffle one feature, importance = loss of fit, works with any model (§4.1 `:203`, §5.1 `:229-241`) — `:362-402`, `:405-462`, `:194-197`.
4. Gradient importance = mean |∂output/∂input| per feature, PyTorch only (§4.1 `:204`, §6.1 `:326-334`, §6.2 algorithm) — `:465-499`; restriction `:199-202`, `:286-289`.
5. Values normalised to sum to 1 with a zero-total guard; permutation uses absolute values (§5.2 `:281-284`, §6.2 `:374-377`) — `:320-325`, `:351-357`, `:396-400`, `:456-460`, `:494-497`.
6. Charts are interactive Plotly (§9.1 library column for bar/radar/dashboard; §1.3 "HTML (interactive)") — every chart function imports `plotly.graph_objects` (`:63`, `:169`, …, `:1098`).
7. An LLM-assisted interpretation exists, with a fallback (§1.3 `:56`, §10) — `feature_importance_ai_interpreter.py:172-205`, `:116-169`. (Which LLM: D2.)
8. A top-10 share above 95 % is flagged as possible circular/leaky features (§14.2 `:1025-1028`) — `feature_importance_ai_interpreter.py:135-138`.
9. Step 5 writes `best_model.*` and `best_model.meta.json` (§2.1 `:68-69`) — `models/model_factory.py:29-32`; host `models/reinforcement/`. (Extensions: D5.)
10. `--save-all-models` / cross-model importance comparison is a TODO (§11.2 heading `:863`) — still not built: no such flag in `meta_prediction_optimizer_anti_overfit.py` argparse (`:3036-3105`); it exists only in `backups/step6_restoration_20251223_213936/` (`:928`).

**UNVERIFIED (1)**
- **U1** The extraction module works at runtime with the four families' saved Step 5 objects (§1.3 `:54`, §11.1). On
  reading, raw Boosters have no `feature_importances_` and no `get_booster`, so they would route to `sklearn_other` →
  sklearn `permutation_importance` (`:239-246`, `:383-392`), and the NN checkpoint is a dict
  (`models/wrappers/neural_net_wrapper.py`, Ch13 B15 blocker 3). Would verify: loading each `best_model.*` and calling
  `get_feature_importance()` — a run, outside this pass.

**Absorbed (6), not counted separately**
- **X1** Importance is available for all four families after Step 5 training (§1.3 `:54` capture aspect; §2.1) → **Ch14 B4** (trees never captured), **B18** (NN gradients never captured), **B5** (`f{i}`/`feature_{i}` keys). Host: 4 of 223 diagnostics files carry importance, all CatBoost.
- **X2** The sidecar is the input to post-training importance work (§2.1 `:69-71`) → **Ch14 Addendum 2 / S10 (G5)**: TB-approved, owner-held; sidecar has no `feature_importance`; `chapter_13_diagnostics.py:449` reads `{}`.
- **X3** SHAP "local explanations — why did THIS prediction happen?" (§7.1 `:427`) → per-survivor attribution, `per_survivor_attribution.py:49-286` (backends `:49-272`, dispatch `:281-286`), cannot execute (**Ch13 B15**) and is ruled blocked (**Ch14 S2**); autonomous use not authorized (**Ch14 S4**).
- **X4** Circular-feature set `{score, confidence, exact_matches, residue_1000_match_rate}` excluded from X; "circular feature detection prevents overfitting" (§3.3 `:179-193`, §5.4 `:306-318`, §16 `:1083`) → **Ch6 D5** (only `score`, `confidence` excluded, `mpo:1482`; no `CIRCULAR_FEATURES` constant in tracked `*.py`).
- **X5** "Check training target is holdout_hits" / "y = holdout_hits, not score" (§10.2 `:818`, §14.1 `:1012`, §14.2 `:1029`) → **Ch6 S1** (`y = holdout_quality`, `mpo:1481`).
- **X6** Category "predictive value" column (§3.2 `:169-177`) and every example ranking led by `intersection_weight`, `skip_entropy`, `lane_agreement_8`, `bidirectional_selectivity` (§5.3, §10.2, §13.1) → **Ch14 Q5** (13 trial-level NPZ fields), **Ch14 D11**, **Ch6 S2**; `skip_entropy` is a 0.0 placeholder (`survivor_scorer.py:783-791`).

---

## 10. Count reconciliation

| Bucket | Items | Count |
|---|---|---|
| Accurate | §9 items 1–10 | 10 |
| BROKEN | B1–B4 | 4 |
| SWITCHED-OFF-BY-RULING | — | 0 |
| DESCRIPTION-ONLY | D1–D12 | 12 |
| UNVERIFIED | U1 | 1 |
| Absorbed into an existing item | X1–X6 | 6 |
| **Total claims checked** | | **33** |

10 + 4 + 0 + 12 + 1 + 6 = **33**. No UNVERIFIED item was resolved during the pass, so no item moved bucket. Note N1
(§6) is outside the chapter's text and is not a claim.

---

## 11. Updates needed elsewhere (listed, not made)

- `docs/PROPOSAL_Addendum_H_Feature_Importance_Integration.md:6`, `:24-27`, `:60-70` — "IMPLEMENTED" for Step 5 is
  no longer true since `e0b55a1`/`249052b` (B1). Proposal text; record, do not edit.
- `docs/CHAPTER_14_TRAINING_DIAGNOSTICS.md:164` "Chapter 11: Post-training feature importance (permutation, gradient,
  SHAP)" and `docs/CHAPTER_3_SCORER_META_OPTIMIZER.md:1325` "Permutation, gradient, SHAP methods" (B2). Both chapters are
  closed for this pass; for their next revision.
- `agent_manifests/definitions.json:178` example `feature_count: 62` (D6). Gitignored host file.
- `feature_importance_ai_interpreter.py:238`, `:260`, `:263` — prompt asks "how the PRNG can be predicted" (D12 framing).
  Code; a separate brief.
- `feature_importance.py:95-119` and `feature_drift_tracker.py:197-221` — the 60-name lists (B4). Code; a separate brief.
- `docs/PROJECT_FILE_CATALOG.md` Chapter 11 row and §7 gap 6 (Chapter 11 now audited); `docs/BACKLOG.md` §1 if it
  names Chapter 11 as unscrutinised.
- The `323b6fa` "Doc sync" revert of `49e8ad4` (N2) is a third instance of the stale-copy overwrite class the fact file
  names (`docs/TFM_PROJECT_FACTS_SKILL.md` §2.7 #2; Chapter 2 §0.1). The same commit touched 23 other files, including
  `docs/CHAPTER_3_SCORER_META_OPTIMIZER.md`, `docs/CHAPTER_6_ANTI_OVERFIT_TRAINING.md`, `docs/CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md`,
  `docs/CHAPTER_12_WATCHER_AGENT.md` (−81 lines); whether any of those changes were also reverts was not checked here.
  Candidate entry for the skill's defect list.

---

## 12. Files opened (execution proof)

`docs/CLAUDE_CODE_INSTRUCTIONS_CHAPTER_CORRECTIONS_20261111.md`; the common brief (scratchpad);
`docs/CHAPTER_11_FEATURE_IMPORTANCE_VISUALIZATION.md`; `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md`; `docs/CHAPTER_13_AUDIT.md`;
`docs/CHAPTER_14_AUDIT.md`; `docs/CHAPTER_6_AUDIT.md`; `docs/CHAPTER_5_AUDIT.md`; `docs/CHAPTER_5_ML_ARCHITECTURE_OPTIMIZER_v2.md`;
`docs/PIPELINE_BEHAVIOUR_MODEL.md`; `docs/PROPOSAL_Addendum_H_Feature_Importance_Integration.md`;
`docs/S111_TEAM_BETA_BRIEFING.md`; `docs/SESSION_CHANGELOG_20260225_S111.md`; `docs/S172_ATTRIBUTION_AND_FEATURE_TRACE_REPORT.md`;
`docs/PATCH_Chapter11_LLM_Update_v2.md`; `docs/apply_chapter11_patch.sh`; `docs/TODO_CONSOLIDATED.md` (grep);
`docs/TFM_PROJECT_FACTS_SKILL.md` (`:2850-2866`); `docs/PROJECT_FILE_CATALOG.md` (grep); `~/.claude/skills/lead-handling/SKILL.md`;
`feature_importance.py`; `feature_visualizer.py`; `feature_drift_tracker.py`; `feature_importance_ai_interpreter.py`;
`feature_importance_interpreter.py`; `web_dashboard.py`; `per_survivor_attribution.py`; `meta_prediction_optimizer.py`;
`meta_prediction_optimizer_anti_overfit.py`; `survivor_scorer.py`; `full_scoring_worker.py`; `chapter_13_diagnostics.py`;
`train_single_trial.py`; `utils/canonical_records.py`; `models/model_factory.py`; `git show a15b524:agents/watcher_agent.py`;
`llm_services/llm_server_config.json`; `agent_manifests/definitions.json`; host artifacts `feature_importance_step4.json`,
`feature_importance_step5.json`, `feature_drift_step4_to_step5.json`, `survivors_with_scores.json` (first record keys),
`models/reinforcement/best_model.meta.json` (keys), `diagnostics_outputs/` (listing, key search),
`~/dashboard_work/DESIGN_DASHBOARD_MINER_v1.md` (grep).
