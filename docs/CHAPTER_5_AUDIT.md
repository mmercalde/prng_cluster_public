# Chapter 5 audit — Step 4, the Adaptive Meta-Optimizer (capacity planner)

Repo: `/home/michael/distributed_prng_analysis` on VM 101 (`192.168.3.177`), user `michael`, read-only.
Code base: `a15b5247336255801de8cf67d52653701b1e8c80` (`a15b524`). Working-tree HEAD is `741822b` (batch-1
documentation commit); `git diff --name-only a15b524 741822b` touches only `docs/`, so the code read at
`741822b` is the code at `a15b524`. **Exception found during the pass:** another session modified
`agents/watcher_agent.py` (and `agents/watcher_dispatch.py`) in the working tree while this audit ran
(`git status`: ` M`; +86/−2 against `a15b524`). Every `agents/watcher_agent.py` anchor below was therefore
re-checked against `git show a15b524:agents/watcher_agent.py`, and all of them hold at `a15b524`. No other file
cited here differs from `a15b524` (`git diff --stat a15b524 -- <files>` is empty).
Audited: `docs/CHAPTER_5_ML_ARCHITECTURE_OPTIMIZER_v2.md` (423 lines at `a15b524`) against the code.
Authority: `docs/CLAUDE_CODE_INSTRUCTIONS_CHAPTER_CORRECTIONS_20261111.md` (REV1), §0, §1, §3, §5.
Method: governance trail first, then chapters, then code, then host state. Every anchor below was opened in this
session. Nothing was executed except read-only `python3 -I` one-liners over host JSON artifacts and one piece of
pure arithmetic (the architecture scores in B1). No training, no Optuna, no pipeline step was run.
Numbering: bare "Step N" means the **executable** scheme (`docs/PIPELINE_BEHAVIOUR_MODEL.md` §1.1;
`agents/watcher_agent.py:387-395` `STEP_SCRIPTS`). Chapter 5 documents executable **Step 4**.
Framing held throughout: TFM is functional mimicry of surface output; no finding here assumes a "true seed".

---

## 1. Verdict

Null results first:

- **Step 4's output reaches no consumer that reads its values.** Step 5 never opens
  `reinforcement_engine_config.json`; WATCHER uses the file only as an existence and freshness precondition for
  Step 5 (B5). Denominator: the Step 5 script, its CLI, its subprocess modules, and the Step 5 manifest were all
  searched.
- **Every Step 4 run on the host gave the same answer.** 27 of 27 archived results
  (`optimization_results/meta_optimization_*.json`, 2025-11-30 → 2026-03-07) give architecture `[256, 128, 64]`,
  150 epochs, confidence 0.245, survivor count 475 or 476. The code explains this: the primary source falls back
  to constants (S1, governed), the architecture choice depends on no input (B1), and the feedback source has no
  producer (B4).

Counts (41 claims; full reconciliation in §10):

- Accurate: **19**
- BROKEN: **5**
- SWITCHED-OFF-BY-RULING: **1**
- DESCRIPTION-ONLY: **10**
- UNVERIFIED: **1**
- Absorbed into another item, not counted separately: **5**

What matters most:

1. The design principle holds. Step 4 does **not** read survivor-level or holdout data
   (`adaptive_meta_optimizer.py` opens only the config, the window results, the history, the feedback file, and
   its own outputs: `:167`, `:226`, `:321`, `:482`, `:972-1078`).
2. The 60 % "primary" source is a constant. That is **governed** (TB Q4, "preserve the drift",
   `docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md:304`, `:723`) and is not reported as new.
3. Four other chapter claims fail without any ruling behind them: the architecture heuristic (B1), two of the
   three history metrics (B2, B3), the feedback source (B4), and Step 5 consuming the output (B5).

---

## 2. Numbering, canonical version, manifests, and fact-file statements

| # | Question | Result | Evidence |
|---|---|---|---|
| N1 | Which WATCHER step is this chapter? | **Executable Step 4** | `agents/watcher_agent.py:392` `4: "adaptive_meta_optimizer.py"`; `:403` `4: "ml_meta.json"`; `STEP_NAMES[4]` "ML Meta-Optimizer"; `agent_manifests/ml_meta.json:4` `"pipeline_step": 4`; `docs/PIPELINE_BEHAVIOUR_MODEL.md:102`, §6 (`:494`). The chapter's title "(Step 4)" uses the executable scheme. |
| N2 | Other versions of Chapter 5? | **None.** `_v2` is the only version and is canonical | `ls docs/CHAPTER_5*` → one file; `git log --all --format= --name-only` over all history → only this path; `find / -xdev -iname '*chapter_5*'` → this file plus the session baseline copy. The "Version 1.0.0 – December 30, 2025" in the chapter's own history was never committed (U1). |
| N3 | Related non-chapter text | `agent_contexts/step4_adaptive_meta.md` (WATCHER mission context; repeats §3.1 of the chapter); root-level `ml_meta.json` (**untracked**, `git ls-files` empty; v1.0.0 manifest that still describes Optuna and `--input survivors_with_scores`, `:84`). WATCHER reads `agent_manifests/` (`agents/watcher_agent.py:776`), not the root copy. Neither edited. |
| N4 | `agent_manifests/definitions.json` (gitignored, `.gitignore:41`) | **No Step 4 entry.** 220 lines, schema 1.1.0, `updated_at` 2026-01-02; `pipeline_steps` holds `step5_anti_overfit` and `step6_prediction` only (`:6-97`). `/bin/grep -n -i "step4\|step 4\|adaptive\|ml_meta\|reinforcement_engine_config"` → 0 hits. Never in git history (`git log --all -- agent_manifests/definitions.json` empty). |
| F1 | Fact file `TFM_PROJECT_FACTS_SKILL.md:164`: Step 4 → `reinforcement_engine_config.json` (capacity only) | Accurate | `adaptive_meta_optimizer.py:994-1053` |
| F2 | Fact file `:1095`: "Supervised model · Steps 4–5 · 89 survivor-derived features" | Wrong for Step 4 (no supervised model in Step 4). Not edited here; listed in §11 | `adaptive_meta_optimizer.py:740-792` (no model is trained) |

---

## 3. What actually works end to end today

1. WATCHER runs Step 4 locally (`agent_manifests/ml_meta.json:30` `"distributed": false`) with
   `--mode full --window-results optimal_window_config.json --lottery-data train_history.json --apply`, built from
   `default_params` (`:104-109`); a `True` boolean becomes a bare flag (`agents/watcher_agent.py:2143-2148`).
2. The script loads `optimal_window_config.json` and `train_history.json` (`adaptive_meta_optimizer.py:226`,
   `:321`), computes a survivor count, an architecture and an epoch count (`full_calibration`, `:838-891`), writes
   `optimization_results/meta_optimization_<ts>.json` and `meta_optimization_results.json` (`:1057-1081`), and
   with `--apply` in full mode merges into `reinforcement_engine_config.json` (`:1171-1172`, `:994-1053`).
3. WATCHER's success test is file existence of `reinforcement_engine_config.json`
   (`agent_manifests/ml_meta.json:37-39`, `:122`).
4. Step 5 then requires that file to exist and to be older than Step 5's output
   (`agent_manifests/reinforcement.json:6-10`; `check_output_freshness`, `agents/watcher_agent.py:803-834`).
   **Nothing reads the values** (B5).

Host state (point-in-time, 2026-11-11): `reinforcement_engine_config.json` is **absent** (root and
`find / -xdev`); `optimal_window_config.json` is **absent**; `train_history.json` present (list of 14,454 ints,
2026-08-21); no `optimization_results/reinforcement_feedback.json`. Last Step 4 run on the host: 2026-03-07.

---

## 4. BROKEN

"Governed" = already recorded in the governance trail. None of B1–B5 was found there.

**B1 — The architecture choice depends on no input; it always selects `[256, 128, 64]`.** *(not found in the governance trail)*
- Chapter: "Heuristic based on problem complexity", `[128, 64, 32]` "Complex (default)" (§5.2, `:205-213`); example output `[128, 64, 32]` (§6.1, `:242`).
- Code: `optimize_network_architecture` (`adaptive_meta_optimizer.py:740-792`) scores each option by its own size only: `capacity = min(1, sum(arch)/256)`, `simplicity = 1 - len(arch)/4`, `score = 0.7·capacity + 0.3·simplicity` (`:765-771`). The `survivor_count` argument is only logged (`:752`). The code calls this a simulation (`:757`, `:764`).
- Arithmetic over the configured options (`adaptive_meta_optimizer_config.json` `optimization.network_architecture.options`, same as `:120-125`): `[32]` 0.3125, `[64,32]` 0.4125, `[128,64,32]` 0.6875, `[256,128,64]` **0.775** → always the last.
- Host: 27 of 27 archived results show `[256, 128, 64]`.
- Repair would have to: make the selection depend on an input the chapter names (problem complexity), or state that the architecture is fixed by the option list.

**B2 — Entropy is computed and thrown away.** *(not found in the governance trail)*
- Chapter: "Entropy — higher entropy = more complex = more capacity needed" (§4.3, `:170`).
- Code: `estimate_min_survivor_count` (`:345-373`) computes normalised entropy; its result `pattern_min` is logged (`:648`, `:651`) and enters nothing in `result` (`:713-728`). The pool minimum comes from the window range and the multiplier (`:705`).
- Repair would have to: feed the entropy estimate into the derived parameters, or state that it is a log line only.

**B3 — Regime-change frequency is never computed.** *(not found in the governance trail)*
- Chapter: "Regime change frequency — frequent changes need adaptability" (§4.3, `:172`).
- Code: `get_regime_change_frequency` is defined (`:415-454`) and called nowhere (`/bin/grep -n get_regime_change_frequency` → the definition only). `handle_regime_change` (`:946`) is not reachable from the CLI (`:1156-1165`).
- Repair would have to: call it from the derivation, or state that it is unused.

**B4 — The reinforcement-feedback source has no producer.** *(not found in the governance trail)*
- Chapter: `reinforcement_feedback.json` from "Step 6 (optional)" (§2.2, `:84`); "Read reinforcement feedback — Learn from past runs" (§3.1, `:110`); weight grows 5 % → 25 % with confidence (§4.4, `:174-179`).
- Code: the reader expects `optimization_results/reinforcement_feedback.json` (`:471`), a list of records with `performance`, `survivor_count`, `network_architecture`, `training_epochs` (`:502-510`). The weighting logic exists (`:554-563`).
- Surfaces searched for a writer: all tracked `*.py`/`*.sh` outside `backups/` (`/bin/grep -rn reinforcement_feedback`: only this reader); tracked and untracked `*.json`/`*.md`/`*.txt` (config, docs, and the archived results that name the weight); git history (`git log --all -S"reinforcement_feedback.json"`: `0df411a`, `c9f7f07`, `89b6087`, `53a3829`, `d14dcdd` — reader, backups, the chapter and the mission-context doc; no writer); host (`find / -xdev -name "reinforcement_feedback*.json"`: none). Step 6 (`prediction_generator.py`) does not write it.
- Consequence on reading: the source is always "not yet available" (`:612`, `:668-671`). Its base weight 0.05 stays in the denominator with no value in the numerator (`:686-699`), so the derived optimum is 5 % below the weighted mean of the other two.
- Repair would have to: name one producer that writes the schema the reader expects, or state that the source is not connected.

**B5 — Step 5 does not use Step 4's output values.** *(not found in the governance trail)*
- Chapter: `survivor_count` → "Sample size selection", `network_architecture` → "Neural net config", `training_epochs` → "Early stopping baseline", `confidence` → "Adjustment decisions", all "Used By Step 5" (§6.2, `:266-271`); output "Consumed By Step 5" (§2.4, `:98`).
- Code: `meta_prediction_optimizer_anti_overfit.py` contains no reference to `reinforcement_engine_config`, `ReinforcementConfig` or `reinforcement_engine` (`/bin/grep -n`); its CLI has no such inputs (`:3040-3105`); NN hidden layers come from Optuna or the default `[256, 128, 64]` (`:908`, `:2710`). `subprocess_trial_coordinator.py`, `train_single_trial.py`, `training_diagnostics.py`, `prediction_generator.py`: no reference. The only coupling is WATCHER's existence/freshness gate (`agent_manifests/reinforcement.json:6-10` → `agents/watcher_agent.py:803-834`). `reinforcement_engine.py:1045` reads the file but is not in `STEP_SCRIPTS`.
- Repair would have to: make Step 5 read the named fields, or state that Step 4's output is an ordering precondition only.

---

## 5. SWITCHED-OFF-BY-RULING

**S1 — The window-optimizer analysis (the 60 % "primary" source) runs on constants.** *(GOVERNED)*
- Chapter: survivor-count range, convergence and stability "from Step 1 results" (§4.2, `:160-165`); Step 4 decides "based purely on how the sieve optimization performed" (§2.1, `:74-76`).
- Code: `WindowOptimizerAnalyzer` reads `best_result.bidirectional_count`, `best_result.precision` and `all_results[]` (`adaptive_meta_optimizer.py:244-263`, `:271-295`). Step 1 writes a flat file with a top-level `bidirectional_count` and no `best_result` / `all_results` (`window_optimizer.py:951-965`; host `optimal_window_config_test.json` keys confirm). So with a non-empty file: optimal 500, min 250, max 1000, confidence 0.0 (`:247`, `:255-257`, `:263`); speed 0.5, stability 0.5 (`:272-273`) → epochs 150 (`:814-819`). Host: 27 of 27 results have confidence 0.245 (= 0·0.60 + 0.7·0.35 + 0·0.05, `:709-711`) and epochs 150.
- Ruling: TB Q4 — "do NOT add `best_result`, `all_results`, or `mod` to `optimal_window_config.json` inside S172. That is a separate compatibility-changing patch. Miner inherits the silent-fallback behavior unchanged." (`docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md:304`; §13 Q4 `:723`; §11.F `:561-567`). Also `docs/PIPELINE_BEHAVIOUR_MODEL.md:519-524`. Owner ruling §0.1 applies: this is a recorded review position the owner may keep or lift.
- Lifting it would enable: Step 4's pool size, epoch count and confidence to vary with what Step 1 actually found (best and per-trial survivor counts, convergence, stability), instead of being the same on every run. Neither recommended nor opposed.
- Note: the governed text quotes the fallback as `{'min':100,'optimal':500,'max':2000}` (`adaptive_meta_optimizer.py:242`). That branch fires only for an empty or missing-content file. For a real Step 1 file the fallback is 250 / 500 / 1000 (above). Listed in §11; the governed documents were not edited.

---

## 6. DESCRIPTION-ONLY

| # | Chapter | Code at `a15b524` |
|---|---|---|
| D1 | "Step 4 of the 6-step pipeline" (§1.1, `:29`) | `STEP_SCRIPTS` has seven entries, 0–6; Step 0 is TRSE (`agents/watcher_agent.py:387-395`) |
| D2 | "62 features per survivor" (§1.3, `:45`) | live sidecar `models/reinforcement/best_model.meta.json` `feature_schema.feature_count` = **89** (2026-03-06); Step 4 does not read it either way |
| D3 | survivor formula `w·0.60 + p·0.35 + f·fw`, `min = opt·0.5`, `max = opt·2.0` (§5.1, `:187-201`) | divides by the sum of weights (`:699`); `min = max(window_min, int(opt·0.5))` (`:705`); with feedback, window and pattern weights each drop by `fw/2` (`:680-684`) |
| D4 | §6.1 JSON is "the config file format" for `reinforcement_engine_config.json`; §2.4 lists only that output | that JSON is `full_calibration()`'s return (`:861-877`), saved to `optimization_results/meta_optimization_results.json` and a timestamped copy (`:1057-1081`). `reinforcement_engine_config.json` receives `model.hidden_layers`, `training.epochs`, `survivor_pool.max_pool_size` and `_meta_optimizer{last_calibration, calibration_type, confidence, weights_used}` (`:1031-1049`). Manifest outputs list both files (`agent_manifests/ml_meta.json:16-19`) |
| D5 | `--lottery-data` Required: Yes (§7.2, `:301`) | not `required=True` (`:1110-1111`); without it the path comes from `adaptive_meta_optimizer_config.json` → `lottery_data.json`, absent on host → `FileNotFoundError` → exit 1 (`:1146-1153`). Required in practice |
| D6 | `--config` default "-" (§7.2, `:303`); table lists five flags | default `adaptive_meta_optimizer_config.json` (`:1101-1103`); `--test` exists and is not listed (`:1114-1128`); `--apply` acts only in `--mode full` (`:1171`) |
| D7 | `--survivor-data` and `--holdout-history` "exist for backward compatibility but are intentionally ignored" (§7.3, `:305-312`) | neither exists. `--survivor-data` existed at `185e755` and was removed at `6e905e7` (2026-01-01, "Step 4 contract fix"); `--holdout-history` existed in no committed version (`0101306`, `c9f7f07`, `185e755`, `a15b524`). `parse_args()` (`:1117`) rejects unknown flags. Legacy callers still pass `--survivor-data`: `complete_whitepaper_workflow.py:198-203`, `complete_whitepaper_workflow_with_meta_optimizer.py:309-314` (not the WATCHER path) |
| D8 | Step 5 "Use holdout_hits for evaluation (FIRST time!)" (§8.1, `:333`; §1.3 `:59`) | Step 5's target is `holdout_quality` (`meta_prediction_optimizer_anti_overfit.py:612`, `:1481`); `holdout_hits` is excluded from features (`:629`). Step 5 is the first step to consume holdout-derived fields. Chapter 6 owns the detail |
| D9 | manifest snippet with `inputs{window_results, lottery_data}`, `outputs{config}`, `next_agent` (§8.2, `:340-355`) | real shape: lists `required_inputs`/`inputs`/`outputs`, `actions[0].args_map`, `follow_up_agents: ["reinforcement_agent"]` (`agent_manifests/ml_meta.json:6-36`). `agent_name`, `pipeline_step` 4 and the script match |
| D10 | "Step 4 is the last step before survivor data is touched" (Summary, `:400`) | Steps 1–3 produce and score survivors; the accurate form is "the last step before survivor-level data is used to train a model". `preflight_check.py:286` lists `survivors_with_scores.json` as a Step 4 input (existence check only; Step 4 does not open it) |

---

## 7. Host questions (instruction §5) and Chapter 14 cross-references

- **`agent_manifests/definitions.json`** — read on the host: no Step 4 entry (§2 N4).
- **Ch 14 B3** (final tree models get no diagnostics): touches this step only through Step 5. Step 4's
  `network_architecture` would configure the NN alone; the tree families take nothing from Step 4 either way (B5).
- **Ch 14 B6** (compare-models summary filename/key mismatch): Step 5 picks the winner without any Step 4 input; the
  health check then judges the wrong file (`docs/CHAPTER_14_AUDIT.md:228`). No Step 4 interaction.
- **Skip registry consumer** `meta_prediction_optimizer_anti_overfit.py:173-174` (inside `_s88_run_compare_models`,
  `:141`; path `diagnostics_outputs/model_skip_registry.json`, `:150`): opened. Host registry: `neural_net`
  `consecutive_critical` 1, `skip_active` false → not skipped (threshold 3). Step 4 neither reads nor writes it.
- **RANGE-MINER / 22-array NPZ / window anchor:** Step 4 reads no NPZ and no `offset`/`window_anchor`/
  `generator_phase` (`/bin/grep -n` on `adaptive_meta_optimizer.py` → 0 hits). The window-anchor change (`69ca910`)
  touched `utils/canonical_records.py`, the miner, and `window_optimizer_integration_final.py`, not Step 4's inputs.
  RANGE-MINER's only effect is §11.F of the proposal: Step 4 output must be identical under the miner, and it is
  identical by construction because both engines omit the same keys (S1).
- Scheduler surfaces: `crontab -l` → none for michael; `systemctl list-units --all` → no unit that runs Step 4.

---

## 8. Surfaces searched, and surfaces unavailable

**Searched and read**
- Governance: `docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md` (§4.4.3, §11.F, §13 Q4), `docs/TB_RULING_S176_WATCHER_KPI.md`
  (Q4 D1, `:600-650`), `docs/WATCHER_KPI_CALIBRATION_FINDINGS_S176.md:198-199`, `docs/SESSION_CHANGELOG_20260720_S176.md:55`;
  `/bin/grep -l -i "step 4|adaptive_meta|ml_meta"` over `docs/TB_RULING_*`, `docs/TEAM_ALPHA_*`, `docs/PROPOSAL_*`,
  `docs/proposals/*` (24 hits; the Step 4 lines of `PROPOSAL_ML_Architecture_Remediation_v2_0.md`,
  `PROPOSAL_Modular_Step_Automation_Framework_v1_1.md`, `PROPOSAL_v1_3_FINAL_ACCEPTANCE.md`,
  `PROPOSAL_EPISTEMIC_AUTONOMY_UNIFIED_v1_3.md` read); `/bin/grep` for `optimize_network_architecture`, `max_pool_size`,
  `256, 128, 64]`, `Option A` + capacity, `survivor-data`. The docstring's "Team Beta Approved – Option A"
  (`adaptive_meta_optimizer.py:6`) matches commit `0df411a` ("Step 4 contract mismatch (Team Beta Option A)"); no
  standalone ruling file for it was found.
- Chapters and models: this chapter (every line); `CHAPTER_2_BIDIRECTIONAL_SIEVE.md` (model, not edited);
  `CHAPTER_13_AUDIT.md` (layout); `CHAPTER_14_AUDIT.md` (B3, B6); `CHAPTER_14_TRAINING_DIAGNOSTICS.md` (§16–§19 form);
  `CHAPTER_6_ANTI_OVERFIT_TRAINING.md` (Step 4 lines `:644-658`); `CHAPTER_12_WATCHER_AGENT.md` (Step 4 lines);
  `PIPELINE_BEHAVIOUR_MODEL.md` §1.1 and §6; `TFM_PROJECT_FACTS_SKILL.md` (`:164`, `:1092-1106`);
  `PROJECT_FILE_CATALOG.md` (`:284`, §7); `BACKLOG.md` §1; `TODO_CONSOLIDATED.md` (no Chapter 5 / Step 4 item).
- Code, read in full: `adaptive_meta_optimizer.py`, `agent_manifests/ml_meta.json`, `adaptive_meta_optimizer_config.json`.
  In the relevant parts: `agents/watcher_agent.py` (`STEP_SCRIPTS`, `STEP_MANIFESTS`, `check_output_freshness`,
  command build `:2080-2200`, `_find_results`), `window_optimizer.py` (`:935-1052`), `meta_prediction_optimizer_anti_overfit.py`
  (imports, `:141-180`, `:610-632`, `:908`, `:1322`, `:1481`, `:2710`, CLI `:3036-3105`), `agent_manifests/reinforcement.json`,
  `agent_manifests/definitions.json`, `preflight_check.py:275-290`, `agents/registry/parameter_registry.py:430-478`,
  `complete_whitepaper_workflow*.py` and `run_complete_pipeline.py` (Step 4 call sites), `agent_contexts/step4_adaptive_meta.md`
  (head), root `ml_meta.json` (head).
- Git history: `git log` of `adaptive_meta_optimizer.py`, of the chapter, of `agent_manifests/ml_meta.json`,
  of `definitions.json`; `git show` of `0101306`, `c9f7f07`, `185e755`, `6e905e7`, `69ca910`; `git log --all -S` for
  `reinforcement_feedback.json` and `holdout-history`.
- Host: `optimization_results/` (27 Step 4 results, `window_optimizer_results.json`), `train_history.json`,
  `optimal_window_config*.json`, `models/reinforcement/best_model.meta.json`, `diagnostics_outputs/model_skip_registry.json`,
  `find / -xdev` for `reinforcement_engine_config*.json`, `reinforcement_feedback*.json`, `*chapter_5*`; `crontab -l`;
  `systemctl list-units --all`.

**Not available, and where it matters**
- **WATCHER run logs** for the 27 Step 4 runs: whether each was WATCHER- or hand-launched is unknown. It does not
  change any label.
- **The `reinforcement_engine_config.json` that `--apply` wrote**: absent on the host now; what removed it is not
  established. The write path is read from code only.
- **Chapter v1.0.0** (U1): not in git history or on the host.
- **ser8 archives, the private remote, rigs**: not consulted (no SSH by brief).

---

## 9. Accurate claims and unverified claims (for the count)

**Accurate (19)**
1. The chapter is executable Step 4; script `adaptive_meta_optimizer.py` (title, `:1-6`) — `agents/watcher_agent.py:392`; `agent_manifests/ml_meta.json:4`, `:23`.
2. Step 4 reads no survivor-level or holdout data (§1.2, §2.3, §3.2, Summary) — `adaptive_meta_optimizer.py` opens only `:167`, `:226`, `:321`, `:482`, `:972`, `:980`, `:1012`; no survivor/holdout path; `:99`.
3. Step 4 does not choose the model type (§3.2, §6.3, §9.2) — no model-type field in `:861-877` or `:1031-1049`; `:887-888`.
4. Step 4 runs no Optuna and no evaluation (§3.2, §9.3) — no optuna import (`:48-66`); the "trials" are a formula (`:757-771`).
5. Inputs are `optimal_window_config.json` and `train_history.json` (§1.3, §2.2) — `:83-93`, `:1140-1143`; `agent_manifests/ml_meta.json:6-9`, `:26-27`; host `train_history.json` is a flat int list, parsed at `:332-333`.
6. Weights 0.60 / 0.35 / 0.05 → cap 0.25 (§4.1) — `:83-101`; `adaptive_meta_optimizer_config.json` `sources`.
7. Feedback weight grows from initial to max with confidence; best past config and confidence are derived (§4.4) — `:491-563`. (No producer: B4.)
8. Temporal stability sets the pattern optimum (§4.3, bullet 2) — `:375-413`, used `:649`, `:691-694`.
9. Epoch formula `100 · 1/(speed+0.5) · (1+(1-stability))`, clamped 50–500 (§5.3) — `:814-824`. (Inputs constant: S1.)
10. `--apply` writes `reinforcement_engine_config.json` (§2.4, §7.2) — `:1171-1172`, `:994-1053`.
11. Output carries no `model_type`, `hyperparameters`, `holdout_metrics` (§6.3) — `:861-877`, `:1031-1049`.
12. §7.1 command is valid and matches what WATCHER sends — `:1104-1113`; `agent_manifests/ml_meta.json:104-109`; `agents/watcher_agent.py:2143-2148`.
13. `--mode` default `full`; `--window-results` defaults (via config) to `optimal_window_config.json` (§7.2) — `:1104-1109`, `:85`.
14. The manifest has no `survivors` or `holdout_history` input (§8.2 note) — `agent_manifests/ml_meta.json:6-15`.
15. Step 5 is the first step to load `survivors_with_scores.json` (§1.3, §8.1) — `meta_prediction_optimizer_anti_overfit.py:3041`; `agent_manifests/reinforcement.json:6-10`.
16. Step 5 selects among `neural_net` / `xgboost` / `lightgbm` / `catboost` (§8.1) — `:3057-3062`, `:148`.
17. Step 5 writes `best_model.*` + `best_model.meta.json` (§8.1) — `:1322`; `agent_manifests/definitions.json:10-13`.
18. Step 5 uses Optuna with cross-validation (§9.3) — `:42-45`, `--k-folds` `:3047`.
19. Pipeline order Step 3 → Step 4 → Step 5 (§1.3, §8.1) — `agents/watcher_agent.py:391-393`.

**UNVERIFIED (1)**
- **U1** Version History: "Version 1.0.0 – December 30, 2025 – Initial documentation (contained incorrect expectations)" (`:417-418`). Not in git history (`git log --all` over every Chapter 5 path) or on the host. Would verify: the ser8 pre-repository archive.

**Absorbed (5), not counted separately**
- **X1** §2.1 "based purely on how the sieve optimization performed" → **S1**.
- **X2** §5.2 "`[128, 64, 32]` Complex (default)" and the §6.1 example values (500 / `[128,64,32]` / 100 / 0.75) → **B1** (architecture) and **S1** (count, epochs, confidence).
- **X3** §2.2 "`reinforcement_feedback.json` · Step 6 (optional)" → **B4**.
- **X4** §3.1 "Step 4 DOES" table (seven rows, each restating items 5, 9, 10, S1, B1, B4) → those items.
- **X5** §9.4 "the `--survivor-data` argument ... is intentionally ignored" → **D7**.

---

## 10. Count reconciliation

| Bucket | Items | Count |
|---|---|---|
| Accurate | §9 items 1–19 | 19 |
| BROKEN | B1–B5 | 5 |
| SWITCHED-OFF-BY-RULING | S1 | 1 |
| DESCRIPTION-ONLY | D1–D10 | 10 |
| UNVERIFIED | U1 | 1 |
| Absorbed into another item | X1–X5 | 5 |
| **Total claims checked** | | **41** |

19 + 5 + 1 + 10 + 1 + 5 = **41**. No UNVERIFIED item was resolved during the pass, so no item moved bucket.

---

## 11. Updates needed elsewhere (listed, not made)

- `docs/PIPELINE_BEHAVIOUR_MODEL.md:513-515` repeats "`--survivor-data` and `--holdout-history` exist ... intentionally ignored" (D7).
- `docs/PIPELINE_BEHAVIOUR_MODEL.md:521` and `docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md:301` give the fallback as `{100, 500, 2000}`; for a real Step 1 file it is 250 / 500 / 1000 with confidence 0.0 (S1 note). The proposal is frozen; record, do not edit.
- `docs/TFM_PROJECT_FACTS_SKILL.md:1095` "Supervised model · Steps 4–5" (F2).
- `docs/PROJECT_FILE_CATALOG.md:284` (Chapter 5 row "unaudited") and §7 gap 6 (Chapter 5 now audited); `docs/BACKLOG.md` §1 (names Chapter 5 as unscrutinised).
- `agents/registry/parameter_registry.py:437-478` describes Step 4 as "Optimize ML model architecture using Optuna" with `--trials`, `--min-layers`, `--max-layers` and modes `quick/full/deep`; none exist in the CLI (`adaptive_meta_optimizer.py:1101-1116`). Code, not changed here; belongs to Chapter 10's pass.
- `agents/watcher_agent.py:2221` expects Step 4 results in `optimal_ml_config.json`, which Step 4 does not write — the S176 "D1" item, ruled not a current defect because the gate uses the manifest `primary_output` (`docs/TB_RULING_S176_WATCHER_KPI.md:600-640`). Chapter 12 territory.
- `docs/CHAPTER_6_ANTI_OVERFIT_TRAINING.md:644-658` repeats that Step 5 uses `reinforcement_engine_config.json` "derived training parameters" (B5). For Chapter 6's pass.

---

## 12. Files opened (execution proof)

`docs/CLAUDE_CODE_INSTRUCTIONS_CHAPTER_CORRECTIONS_20261111.md`; `docs/CHAPTER_5_ML_ARCHITECTURE_OPTIMIZER_v2.md`;
`docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md`; `docs/CHAPTER_13_AUDIT.md`; `docs/CHAPTER_14_AUDIT.md`;
`docs/CHAPTER_14_TRAINING_DIAGNOSTICS.md`; `docs/CHAPTER_6_ANTI_OVERFIT_TRAINING.md` (grep); `docs/CHAPTER_12_WATCHER_AGENT.md` (grep);
`docs/PIPELINE_BEHAVIOUR_MODEL.md`; `docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md`; `docs/TB_RULING_S176_WATCHER_KPI.md`;
`docs/TFM_PROJECT_FACTS_SKILL.md`; `docs/PROJECT_FILE_CATALOG.md`; `docs/BACKLOG.md`; `docs/TODO_CONSOLIDATED.md` (grep);
`~/.claude/skills/lead-handling/SKILL.md`;
`adaptive_meta_optimizer.py`; `adaptive_meta_optimizer_config.json`; `agent_manifests/ml_meta.json`;
`agent_manifests/reinforcement.json`; `agent_manifests/definitions.json`; `agents/watcher_agent.py`; `window_optimizer.py`;
`meta_prediction_optimizer_anti_overfit.py`; `preflight_check.py`; `agents/registry/parameter_registry.py`;
`complete_whitepaper_workflow.py`; `complete_whitepaper_workflow_with_meta_optimizer.py`;
`complete_whitepaper_workflow_with_meta_optimizer_v3.py`; `run_complete_pipeline.py`; `agent_contexts/step4_adaptive_meta.md`;
root `ml_meta.json`; host artifacts `optimization_results/meta_optimization_*.json` (27), `optimization_results/meta_optimization_results.json`,
`optimization_results/window_optimizer_results.json`, `train_history.json`, `optimal_window_config_test.json`,
`optimal_window_config.s162_victory.json`, `models/reinforcement/best_model.meta.json`, `diagnostics_outputs/model_skip_registry.json`.
