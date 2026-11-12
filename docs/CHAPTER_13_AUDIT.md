# Chapter 13 audit — live feedback loop and selfplay

Repo: `/tmp/repos/prng_cluster_public` at `a15b524` (public mirror, single commit, no history).
Audited: `docs/CHAPTER_13_LIVE_FEEDBACK_LOOP.md` (+ `_v1_1`), `docs/CHAPTER_13_IMPLEMENTATION_PROGRESS_v3_9.md`,
`docs/TODO_SELFPLAY_AND_LLM_AUTONOMY.md`, `docs/PHASE_9B2_INTEGRATION_SPEC.md`, against the code.
Method: read-only. Governance trail first, then chapter, then code. Every anchor below was opened in this session.
Nothing was executed; "works" means the code path is complete on reading, not that it was run.
Step numbers are the EXECUTABLE scheme (Step 2 = scorer meta-optimizer; the sieve runs inside Step 1).
Framing held throughout: TFM is functional mimicry of surface output; no finding here assumes a "true seed".

---

## 1. Verdict

- Accurate: **25** claims
- BROKEN: **18**
- SWITCHED-OFF-BY-RULING: **9**
- DESCRIPTION-ONLY: **9**
- UNVERIFIED: **7**

(68 claims in total. Lists in sections 3 to 6 and 9 are numbered so the counts can be checked.)

Three things matter more than the rest:

1. **The loop's two inputs do not connect to the pipeline's outputs in the tracked tree.** Chapter 13 grades
   `prediction_pool.json`; Step 6 writes `predictions/next_draw_prediction.json` in a different shape (B1).
   Chapter 13 appends draws to `lottery_history.json`; Steps 1/3/5/6 read `daily3.json`, `train_history.json`,
   `holdout_history.json` (B2). I found neither recorded in the governance trail. A host-side bridge may exist; it is not visible from a clone.
2. **The retrain triggers are already ruled structurally invalid** (`TB_RULING_S176_WATCHER_KPI.md`), and the
   approved repair (`TB_RULING_S179_IMPLEMENTATION_AUTH.md`) is not in the tracked code (B3, S2).
3. **Selfplay cannot change anything.** It emits the transforms it was given, its promotion path drops them, its
   acceptance gate needs R² ≥ 0.80, nothing requests it, and the outer (sieving) episode has no counterpart
   interface (B10, B11, B13, B14, S4).

---

## 2. The five fact-file statements (`docs/TFM_PROJECT_FACTS_SKILL.md` §2.4, §2.5, §2.1)

| # | Statement | Result | Label | Evidence |
|---|---|---|---|---|
| 1 | `propose_transform_update` is a no-op | **Confirmed, and stronger**: it returns `current_transforms.copy()` and is also not on selfplay's live path | **SWITCHED-OFF-BY-RULING** | Body: `policy_conditioned_episode.py:373-401` (`:398-401`). Only caller `:475`, inside `run_conditioned_episode` (`:408`), which has no caller (searched all tracked `*.py`, `*.sh`). Selfplay imports neither (`selfplay_orchestrator.py:139-147`) and copies the active policy's transforms straight into the candidate (`:977-979`, `:1014`). Ruling: `docs/PHASE_9B2_INTEGRATION_SPEC.md:5` ("Team Beta Approved") and `:393-409` ("intentionally deferred — manual policy tuning should be validated first"); tracker `v3_9:45`, `:144`. |
| 2 | Promotion seam selfplay → Chapter 13 is broken (`chapter_13_acceptance.py:224`) | **Confirmed.** The `:224` anchor is correct at this commit (it is the `class SelfplayCandidate` line) | **BROKEN** | `SelfplayCandidate` fields `:226-237` have no `transforms`, `parent_policy_id`, `fingerprint`; `from_dict` `:240-255` discards them. Selfplay does write them (`selfplay_orchestrator.py:303-307`, `:1011-1014`). `promote_candidate` `:818-861` writes the active policy from the dataclass (`:830-845`) with no `transforms`. A policy with no `transforms` key validates and conditions as a passthrough (`policy_transform.py:663`). Only caller of `promote_candidate` is the CLI (`chapter_13_acceptance.py:1007`). |
| 3 | All Chapter 13 triggers are defensive; no opportunity trigger | **Confirmed**, with one precision: `N_DRAWS` is a cadence counter, not a degradation test | **DESCRIPTION-ONLY** | Every trigger condition: `chapter_13_triggers.py:246-319`. Enum `:60-69`. The one positive-signal flag the diagnostics emit, `NEW_STRONG_SURVIVORS` (`chapter_13_diagnostics.py:631-632`), has no reader anywhere (searched tracked `*.py`). The chapter never specifies an opportunity trigger, so no code path is missing relative to it; the disagreement is with "improves continuously" (chapter `:1092`). Source of the characterisation: `docs/TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md:129-134`. |
| 4 | Selfplay inner loop still scores with R² while the pipeline moved to `holdout_quality` | **Confirmed** | **BROKEN** | Fitness is `max(0, val_r2 − 0.05) − penalties` (`inner_episode_trainer.py:136-159`). Target is `holdout_hits`, then `score` (`selfplay_orchestrator.py:933`). `holdout_quality` does not occur in `selfplay_orchestrator.py` or `inner_episode_trainer.py`. Step 5 trains on `holdout_quality` (`meta_prediction_optimizer_anti_overfit.py:612`, `:1481-1482`). Chapter 13's gate requires `val_r2 ≥ 0.80` and `fitness ≥ 0.50` (`chapter_13_acceptance.py:64-65`, `:743-752`; `watcher_policies.json:95-96`). See B13 for why this is a break and not only a naming lag. |
| 5 | `per_survivor_attribution.py` is implemented, invoked, unreachable, unconsumed; "four blockers" | **Confirmed. All four blockers hold. I found a fifth, upstream of them, and a dead call downstream.** | **BROKEN** (the unreachable half). The unconsumed half is a separate item, S5, SWITCHED-OFF-BY-RULING | See B15. |

---

## 3. What actually works end to end today

Each item is a complete code path on reading. None was run. Soak results in the tracker are U1.

1. **Test/manual draw ingestion.** `synthetic_draw_injector.py` requires both `test_mode` and `synthetic_injection.enabled` (`:51-77`), reads `prng_type` from `optimal_window_config.json` (`:84-100`), appends to `lottery_history.json` and raises `new_draw.flag` (`:311`, `:349`, `:408`). `draw_ingestion_daemon.py` does the same for supplied draws (`:231`, `:265`).
2. **Daemon cycle.** `chapter_13_orchestrator.py`: halt checks `:158-175`, flag check `:215-233`, cycle on flag and flag clear `:509-519`.
3. **Diagnostics file and archive.** `generate_diagnostics` `chapter_13_diagnostics.py:750-884`; write + archive `:891-926`. (What it grades is B1.)
4. **Trigger evaluation and priority selection.** `chapter_13_triggers.py:201-370`; approval request file `:419-448`.
5. **LLM advisory.** Grammar-constrained call with heuristic fallback, archived to `llm_proposals/` (`chapter_13_llm_advisor.py:221-250`, `:299-344`, `:346-436`, `:83`).
6. **Acceptance decision.** ACCEPT / REJECT / ESCALATE and the decision log (`chapter_13_acceptance.py:327-568`, `:626-629`).
7. **Human approval that executes.** Only `agents/watcher_agent.py --approve` (`:3741-3779`). In production the WATCHER daemon notifies and waits (`:3354-3395`); it auto-dispatches only when `test_mode` and `auto_approve_in_test_mode` are both true (`:3317-3352`).
8. **WATCHER request queue.** `watcher_requests/*.json` scanned each daemon cycle (`agents/watcher_agent.py:3187-3189`), routed to `dispatch_selfplay` / `dispatch_learning_loop` (`agents/watcher_dispatch.py:403-548`). `dispatch_learning_loop` runs exactly the listed steps (`:268-283`, `:303`). Works when a request file exists; see B10 for who writes one.
9. **Selfplay inner episode.** Load survivors, condition on the active policy, train three tree models, emit candidate, archive, telemetry (`selfplay_orchestrator.py:481-545`, `:667-819`, `:955-1050`).
10. **Advisor → selfplay, two parameters.** `max_episodes` and `min_fitness_threshold` (`agents/watcher_dispatch.py:472-502`, `:149-158`; `selfplay_orchestrator.py:1182-1187`).
11. **Candidate validation and promotion by CLI.** `chapter_13_acceptance.py:967-1021`. Mechanically complete; subject to B11 and B13.
12. **Downstream-score annotation.** `0.50·hit@20 + 0.30·hit@100 + 0.15·hit@300 + 0.05·coverage`, written to the DB, non-fatal (`chapter_13_orchestrator.py:292-319`). Subject to B1; and see note N1.

---

## 4. BROKEN

"Governed" means a ruling or governed document already records it. That is a status, not a new finding, and is marked so the owner does not receive it twice.

**B1 — What Chapter 13 grades is not what Step 6 writes.** *(not found in the governance trail)*
- Chapter: inputs are `prediction_pool.json` and `confidence_map.json` from Step 6 (`:329-330`).
- Code: diagnostics default to those two root-level files (`chapter_13_diagnostics.py:50-51`), and the orchestrator calls with defaults (`chapter_13_orchestrator.py:282`). A missing file silently becomes an empty pool (`:114`, `:129`). Step 6 writes `predictions/next_draw_prediction.json` (`prediction_generator.py:896-907`) whose `predictions` is a list of ints (`:849`, `:853-856`), while the grader calls `pred.get("value", …)` on each element (`chapter_13_diagnostics.py:243`).
- Consequence on reading: with no `prediction_pool.json`, every cycle grades zero predictions: 0 hits, coverage 0, correlation 0.0 (`:223-234`, `:326`), so `consecutive_misses` rises every draw and the downstream score written back is 0.
- Stops at: `chapter_13_diagnostics.py:114`.
- Surfaces searched for a writer of `prediction_pool.json` / `confidence_map.json`: all tracked files outside `docs/` and `backups/`. Hits are readers or declarations only (`agents/watcher_agent.py:2223`, `agents/pipeline/pipeline_step_context.py:148`, `schemas/results_schema_v1.json:928`). Not searched: host, gitignored files, private repo (U2). `TB_RULING_S177_KPI_GOVERNANCE.md:303-327` discusses this file as the single pool source but does not record that nothing produces it.
- Repair would have to: make one artifact, with one schema, both the Step 6 output and the Chapter 13 input. S179 §2.5 already authorises a ranked unique prediction artifact; the two could be the same file.

**B2 — The appended draw does not reach the steps that are supposed to relearn from it.** *(not found in the governance trail)*
- Chapter: "New draw appended to `lottery_history.json`; Step 3 re-runs with expanded history" (`:429-435`); "History ← History + New Draw, then re-invoke" (`:748-753`).
- Code: Chapter 13 reads and appends `lottery_history.json` (`chapter_13_diagnostics.py:52`, `synthetic_draw_injector.py:36`, `draw_ingestion_daemon.py:45`). The steps read other files: Step 3 `train_history.json` / `holdout_history.json` (`agent_manifests/full_scoring.json:120-121`), Step 5 `train_history.json` (`agent_manifests/reinforcement.json:175`), Step 6 `daily3.json` (`agent_manifests/prediction.json:180`). The train/holdout split is written only by Step 1 (`window_optimizer.py:1100-1102`), which is not in the 3→5→6 loop. The production scraper (`daily3_scraper.py`) contains no reference to `new_draw.flag` or `lottery_history` (searched). No tracked code copies between the two histories.
- Stops at: `agent_manifests/full_scoring.json:120`.
- Not visible: host units; the fact file says an enabled boot-triggered scraper unit exists on the host (`docs/TFM_PROJECT_FACTS_SKILL.md:200-202`). Whether anything there raises the flag or mirrors the file is U2.
- Repair would have to: give the feedback loop and the pipeline one dataset authority. The governed one is the pointer manifest (`docs/TFM_PROJECT_FACTS_SKILL.md:601-606`), and `:641-643` records a Beta ruling that adding a draw does not leave prior scores valid, which bears on how a refresh may be done.

**B3 — "Hit rate < 0.01 over 20 draws" is a one-draw miss detector.** *(GOVERNED: `TB_RULING_S176_WATCHER_KPI.md:55-116`, `:799-801`)*
- Chapter `:493`. Code: `current_hit_rate = exact_hits / pool_size` for the current draw (`chapter_13_diagnostics.py:529-531`), compared once (`chapter_13_triggers.py:268-271`). `hit_rate_collapse_window` (`watcher_policies.json:27`) is read by no trigger code.
- Stops at: `chapter_13_triggers.py:269`.
- Repair: authorised by S179 §8 (rolling/sequential Hit@K, audit-only until a walk-forward baseline exists; no threshold may be chosen yet, S176 `:814-818`). No trace of that implementation in the tracked tree: `kpi_governance`, `audit_only`, `BLOCKED_BY_GOVERNANCE`, `unique_rank`, `manual-governance-override` occur in no tracked `*.py`, `*.json` or `*.sh` (U3).

**B4 — Regime-shift inputs are not what the chapter names.**
- Chapter `:506-510`: window decay > 0.5, survivor churn > 0.4, LLM structural-drift confidence > 0.8.
- Code: `window_decay` is the relative fall in single-draw hit rate against the previous draw, so it is 0 or 1.0 (`chapter_13_diagnostics.py:549-553`). `survivor_churn` is the change in survivor count, with the comment "would need survivor ID tracking" (`:555-559`). `llm_confidence_threshold` (`watcher_policies.json:36`) has no reader; the LLM's `recommended_action` / `retrain_scope` are logged, not used (`chapter_13_orchestrator.py:388`; the approval request carries the trigger's action, `:415`, `:422`).
- Stops at: `chapter_13_diagnostics.py:552`, `:559`. Partly governed: S176 `:248-266` lists churn and decay as having no empirical baseline.
- Repair: define both metrics over identities and windows; either consume or drop the LLM threshold.

**B5 — Two of the three advertised approve commands cannot execute.**
- Code: `Chapter13TriggerManager` is constructed only with `watcher_agent=None` (`chapter_13_orchestrator.py:125-127`, `chapter_13_triggers.py:897-898`). `approve_request` marks the request approved, calls `execute_learning_loop` (`:516-537`), which returns False at the None check (`:603-608`), then marks it failed (`:540`). So `chapter_13_orchestrator.py --approve` (`:1078-1080`) and `chapter_13_triggers.py --approve` (`:923-925`) run nothing. The approval text (`chapter_13_triggers.py:484-485`) and WATCHER's Telegram message (`agents/watcher_agent.py:3379`) both recommend them.
- Stops at: `chapter_13_triggers.py:603`.
- Repair: route those commands to the WATCHER execution path or stop advertising them.

**B6 — An approved "3→5→6" loop runs Step 4 as well.**
- Chapter `:776-778`, and Step 4 is listed as never per-draw (`:174`). Code: the request stores `steps_to_run = [3, 5, 6]` (`chapter_13_triggers.py:450-453`), but both executors call `run_pipeline(min(steps), max(steps))` (`agents/watcher_agent.py:3350-3353`, `:3773`), and `run_pipeline` walks a contiguous range (`:2717`, loop `:2802`).
- Stops at: `agents/watcher_agent.py:3773`. `dispatch_learning_loop` already handles a step list correctly (`agents/watcher_dispatch.py:268-283`).
- Repair: have the approval path execute the list, not the range.

**B7 — Cooldown and the N-draw counter are never reset by a retrain WATCHER executes.**
- Chapter `:802`: "Cooldown periods — prevents thrashing". Code: the reset exists only at `chapter_13_triggers.py:619-621` (unreachable, B5) and `:690-692` (`--execute`). `agents/watcher_agent.py --approve` and the daemon path write nothing to `.chapter13_cooldown_state.json` (searched: the file name and both field names occur only in `chapter_13_triggers.py`). With `last_retrain_at` never set, the cooldown test at `:384` is never true and `runs_since_retrain` only grows, so `N_DRAWS` (`:279-288`) is true on every cycle after the tenth.
- Stops at: `chapter_13_triggers.py:384`.
- Repair: the executor that actually runs the retrain must record it.

**B8 — Survivor reinforcement and decay are not computed.**
- Chapter `:141`, `:146`, `:374-379`. Code: `decay_candidates` is initialised empty and never appended (`chapter_13_diagnostics.py:405`, comment `:419-420`). Hits are matched on `predicted_value` / `value` (`:396`), fields no Step-3 writer produces (`predicted_value` occurs only in three `shadow_*` scripts). `reinforce_candidates` and `decay_candidates` have no reader outside this file.
- Stops at: `chapter_13_diagnostics.py:396`.
- Repair: per-survivor prediction and hit tracking plus a consumer. The governed route for this is the attribution/ranked-artifact work in B15, not a second mechanism.

**B9 — Divergence detection and halt have no code.**
- Chapter `:815-820`. Code: `_create_halt` (`chapter_13_orchestrator.py:241-245`) has no caller; `escalate_on_consecutive_rejects` and `escalate_on_divergence` (`watcher_policies.json:50-52`) have no reader (searched tracked `*.py`). The nearest live behaviour is escalation at ≥ 3 consecutive misses (`chapter_13_acceptance.py:475-478`).
- Stops at: `chapter_13_orchestrator.py:241`.
- Repair: implement the three detectors over the cycle history, or strike the section.

**B10 — Nothing in Chapter 13 ever requests selfplay.**
- Chapter `:1198-1201`: "Diagnostics → Triggers → `watcher_requests/*.json` → WATCHER → Selfplay". Code: `request_selfplay` (`chapter_13_triggers.py:762`) and `should_request_selfplay` (`:819`) have no caller; the diagnostics never emit `SELFPLAY_RECOMMENDED` or `recommended_actions.request_selfplay` (flag list `chapter_13_diagnostics.py:599-667`, actions `:681-687`). The enum comment says "enum only, no auto-dispatch" (`:69`). WATCHER's consumer side is complete (item 8 above).
- Stops at: `chapter_13_triggers.py:762`. Related ruling: S179 §2.1 (`:79-101`) requires a gate inside this function so that under `audit_only` it creates no request; that ruling does not note the function has no caller.
- Repair: a governed producer for the request, behind the S179 gate.

**B11 — Promotion drops the policy.** (Statement 2.) *(GOVERNED: system map `:62`; behaviour model `:679-680`)*
- Chapter `:1203-1216`. Code and anchors in section 2, row 2. Net effect: a candidate carrying transforms T becomes an active policy with none, and the next selfplay run conditions on nothing.
- Stops at: `chapter_13_acceptance.py:240`.
- Repair: carry `transforms`, lineage and fingerprint through `SelfplayCandidate` and `promote_candidate`, and reject a promotion whose fingerprint does not match.

**B12 — Promotion has no real-draw validation.**
- Contract (RATIFIED): "No component may update `learned_policy_active.json` without real-draw validation by Chapter 13" (`docs/CONTRACT_SELFPLAY_CHAPTER13_AUTHORITY_v1_0.md:106-110`); validation phase is shadow evaluation of candidate vs active on real draws (`:218-222`). Code: `validate_selfplay_candidate` checks source, status, fitness, `val_r2`, train/val gap and survivor count only (`chapter_13_acceptance.py:733-764`). "shadow" occurs in none of `chapter_13_*.py`, `agents/watcher_dispatch.py`, `selfplay_orchestrator.py`.
- Stops at: `chapter_13_acceptance.py:780`.
- Repair: a shadow comparison over graded draws before `promote_candidate` may write. Depends on B1.

**B13 — Selfplay's objective is R², and the gate it must pass is set where R² cannot reach.** (Statement 4.) *(GOVERNED as "Open": fact file `:347-348`)*
- Anchors in section 2, row 4. The fact file records measured R² of 0.000155 (`:344`; U6, not re-measured). At any R² below 0.05 the fitness is ≤ 0 (`inner_episode_trainer.py:148-157`), so neither `fitness ≥ 0.50` nor `val_r2 ≥ 0.80` can pass, and `--promote` re-validates before writing (`chapter_13_acceptance.py:997-1005`). Emission is ungated (`configs/selfplay_config.json:3`, `min_fitness_threshold: -1.0`), so candidates are written and cannot be accepted.
- The contract lists validation R² as an allowed proxy reward (`docs/CONTRACT_SELFPLAY_CHAPTER13_AUTHORITY_v1_0.md:146-160`), so R² was the approved design in January 2026. I found no ruling that it should stay after the pipeline moved to `holdout_quality`, which is why this is BROKEN and not SWITCHED-OFF.
- Stops at: `inner_episode_trainer.py:148` and `chapter_13_acceptance.py:751`.
- Repair: one objective for the inner loop and the gate, consistent with Step 5's target. Note the Beta condition that `holdout_hits` is a label only, "forbidden as a filter, weight, mask, window" (system map `:216`); `policy_transform.py:48` forbids it only as a mask.

**B14 — Selfplay explores nothing.**
- Contract: selfplay runs outer episodes (GPU sieving) and explores parameter space (`…AUTHORITY_v1_0.md:60-65`). Code: outer episodes are off by default "until coordinator interface is ready" (`selfplay_orchestrator.py:194-196`; `configs/selfplay_config.json:8`). The command it would run is `coordinator.py --episode-id …` and it parses a `SURVIVORS_PATH:` line (`:626-630`, `:652-655`); `coordinator.py` and `scripts_coordinator.py` contain no occurrence of "episode", and nothing prints `SURVIVORS_PATH`. Inside a run, every episode uses the same survivors file and the same policy, loaded once (`:498-499`, `:563`, `:731`).
- Stops at: `selfplay_orchestrator.py:606-607`.
- Repair: an outer-episode interface that exists on both sides, and something that varies between episodes. The second half is S4.

**B15 — Attribution cannot execute.** (Statement 5.) *(Blockers 1 to 4 GOVERNED: system map `:89-101`. Blocker 0 and the dead call: not found in the trail.)*
- Tracker: "Per-Survivor Attribution — COMPLETE" (`v3_9:65`), "Real SHAP attribution computed" (`:104`).
- **Blocker 0 (new) — the gate that starts it never opens.** `_detect_hit_regression` returns True only for a summary flag containing both "hit_rate" and "drop", or for top-level `hit_at_20` < `previous_hit_at_20` (`chapter_13_orchestrator.py:547-557`). `generate_diagnostics` emits neither: the complete flag vocabulary is `chapter_13_diagnostics.py:599-667` and the output keys are `:853-882`. So `hit_regression` at `:325` is always False for real diagnostics. The trace report classifies this gate "IMPLEMENTED AND WIRED" (`docs/S172_ATTRIBUTION_AND_FEATURE_TRACE_REPORT.md:626`) and the system map lists hit-regression detection as WIRED (`:50`); neither checks whether its inputs are produced.
- **Blocker 1 — no producer for `predictions/ranked_predictions.json`.** Reader `chapter_13_orchestrator.py:875-937`, called `:328`. Only other occurrence in tracked code is a test constant (`test_phase_8_soak.py:60`).
- **Blocker 2 — wrong sidecar key.** Reads top-level `feature_names` (`:582`); the sidecar nests it under `feature_schema` (`meta_prediction_optimizer_anti_overfit.py:2903`; `models/model_factory.py:233`). A fallback recovers names from tree model objects (`:624-635`), so this now blocks the neural net only; without names the analysis returns None (`:680-683`).
- **Blocker 3 — bypasses the canonical NN loader.** `torch.load(...)` then `.eval()` (`:596-598`); the checkpoint is a dict (`models/wrappers/neural_net_wrapper.py:375-383`).
- **Blocker 4 — NN attribution omits the training scaler.** Features go straight into the model (`per_survivor_attribution.py:97-101`); Step 6 applies the stored scaler first (`prediction_generator.py:814-818`).
- **Dead call downstream.** On `regime_shift` the code calls `StrategyAdvisor().request_diagnostics_analysis(...)` (`chapter_13_orchestrator.py:840-845`). No such method is defined anywhere in tracked `*.py`; the `AttributeError` is swallowed at `:854-855`.
- Stops at: `chapter_13_orchestrator.py:326`.
- Repair: per system map `:255-256` (P0-C). In addition the gate needs an input that exists. The order is constrained by ruling; see S6.

**B16 — Test-mode "orchestrator route" labels a cycle auto-executed and runs nothing.**
- `docs/WATCHER_POLICIES_REFERENCE.md:105`: orchestrator "auto-executes internally". Code: the branch logs and sets `outcome = "auto_executed_test_mode"`; there is no call after it (`chapter_13_orchestrator.py:410-412`). With `v1_approval_required.retrain_execution` false the same function sets `"auto_execute_disabled"` and also runs nothing (`:419-425`). Bears on the tracker's "73 auto-executed" (`v3_9:55`, U1).
- Stops at: `chapter_13_orchestrator.py:412`.
- Repair: execute, or rename the outcome.

**B17 — The frozen-parameter check covers eight names.** (Latent: nothing is applied today, S3.)
- Chapter `:658`: reject proposals that "touch frozen components (Steps 1, 2, 4)". Code: `FROZEN_PARAMETERS` is eight literal strings (`chapter_13_acceptance.py:72-84`); the grammar lets the LLM name any parameter (`agent_grammars/chapter_13.gbnf`, `param-kv … string`). `forward_threshold`, `offset`, `skip_min` are Step-1 parameters and are not in the set. The vocabulary file that was to supply the list, `chapter_13_parameter_vocabulary.py` (TODO B3), is not in the tracked tree.
- Stops at: `chapter_13_acceptance.py:388`.
- Repair: derive the frozen set from the manifests and the reserved-authority list.

**B18 — Two of the four execution preconditions are not checked where the chapter puts them.**
- Chapter `:190-205`: Step 6 complete, new draw, fingerprint changed, no halt. Code: the orchestrator checks halt and flag only (`chapter_13_orchestrator.py:501-519`). A fingerprint check exists in the ingestion daemon (`draw_ingestion_daemon.py:101`). No Step-6-completion check exists in `chapter_13_orchestrator.py`.
- Stops at: `chapter_13_orchestrator.py:510`.
- Repair: check that a prediction artifact for the draw being graded exists. Same artifact as B1.

**Note N1 (not counted; outside the chapter's text).** `best_rank` is the minimum rank over exact hits *and* predictions within ±5 of the draw (`chapter_13_diagnostics.py:246-258`), and the S140b hit@20/100/300 are derived from `best_rank` (`chapter_13_orchestrator.py:306-309`). So a near value counts as a hit in the score written back. Separately governed: with a 20-entry pool the three tiers collapse to one event (`TB_RULING_S177_KPI_GOVERNANCE.md:303-325`).

---

## 5. SWITCHED-OFF-BY-RULING

Authority changed on 2026-11-11: the owner decides whether each stands. Facts only below.

**S1 — Executing a retrain without human approval.**
- Ruling: chapter `:474-482` and `:1032-1048` (Extension #3, "Deferred"; version note `:1114-1115` attributes the extension set to Team Beta); `docs/PIPELINE_BEHAVIOUR_MODEL.md:954`, `:961-965`; `TB_RULING_S179_IMPLEMENTATION_AUTH.md:525-532` ("no autonomous retraining enabled"). Code agrees: `requires_approval=True` (`chapter_13_triggers.py:369`); tracked policy `test_mode: false`, `retrain_execution: true` (`watcher_policies.json:7`, `:59`).
- If lifted: a fired trigger could run Steps 3→5→6 (or 1→6) unattended.
- What else must change: there is no production auto-execute path to switch on. Setting `retrain_execution` false yields a label, not a run (B16); WATCHER's auto-dispatch is conditioned on `test_mode` (`agents/watcher_agent.py:3317-3331`). Then B1, B2 (what is graded and relearned), B3/B4 (what the triggers mean), B6, B7.

**S2 — KPI-driven triggers must be audit-only until calibrated (BOOTSTRAP).**
- Ruling: `TB_RULING_S176_WATCHER_KPI.md:795-821` (approved / not approved lists); `TB_RULING_S179_IMPLEMENTATION_AUTH.md:513-547`.
- State: the ruling is not reflected in tracked code (B3). Today these triggers still write `pending_approval.json` and notify; the human gate (S1) is the only thing between them and a run.
- If lifted (triggers declared active as they are): they would request retrains on their present definitions. Beta's arithmetic for those definitions, at the uniform Top-20 null: collapse first fires after an expected 1.02 draws, five-miss after 5.31 (`S179:57-65`).
- What else must change: nothing in code is holding them back, so lifting changes no behaviour until S1 is also lifted. The ruled repair needs a walk-forward baseline, which S179 `:546` ties to S172 Phase 7 scheduling.

**S3 — Applying an LLM parameter proposal.** (Chain D.)
- Ruling: Beta, as quoted in `docs/CLAUDE_CODE_INSTRUCTIONS_CHAIN_C_TRUTHFULNESS_HOTFIX.md:104-106` ("`pending_approval` is a valid authority boundary … Calling it before physical application would be worse"); request at `docs/TEAM_ALPHA_AUTONOMY_CONTROL_SURFACE_SUBMISSION.md:70-77`, `:146-148`. The ruling text itself is not a standalone file in this clone. Chapter agrees (`:530-536`).
- Code: accepted proposals end in a string or an approval request (`chapter_13_orchestrator.py:398-441`); `record_applied_changes` (`chapter_13_acceptance.py:635`) has no caller.
- Consequences that follow from the ruling, not separate defects: the reversal and per-parameter cooldown rules (chapter `:654`, `:668`) are vacuous because the change history is never written (`chapter_13_acceptance.py:176-188`, `:419-451`) and `increment_run` (`:663`) has no caller; `parameter_change_history.json` (chapter `:724`) is never produced; chapter §4.4 does not occur.
- If lifted: accepted, low-risk proposals could change parameters.
- What else must change: an apply function (none exists; TODO B12), a location and bound for each parameter (the grammar's parameter is a free string; B17), calls to `record_applied_changes` and `increment_run` after physical application, and an observable effective value. `watcher_policies.json:63` `parameter_application` has no reader. Human-only reserved authority is listed at `docs/PIPELINE_BEHAVIOUR_MODEL.md:989-991`.

**S4 — Selfplay proposing its own transform updates (Phase 9B.3).** (Statement 1.)
- Ruling: `docs/PHASE_9B2_INTEGRATION_SPEC.md:5`, `:393-409`; tracker `v3_9:45`, `:144`. Constraints on any future version: system map `:285` (fields must be survivor-local and inference-time available; global context is not a per-survivor filter space).
- If lifted: `propose_transform_update` could return changed transforms.
- What else must change: selfplay's live path does not call it (`selfplay_orchestrator.py:977-979`); the policy is loaded once per run (`:498-499`); promotion drops transforms (B11); the gate is unreachable (B13); nothing requests selfplay (B10).

**S5 — Acting on attribution.**
- Ruling: tracker `v3_9:232-238` (Team Beta, S85: observe-only before trigger authority); system map `:222-228` (root-cause classification stays observe-only, with the evidence required first) and `:277-278` ("Attribution-driven autonomous parameter changes — Not authorized"; "Observe-only attribution→selfplay prior experiment — Authorized after P0 activation"). Code states it: `chapter_13_orchestrator.py:321-323`.
- If lifted: the root-cause diagnosis could feed trigger evaluation or selfplay priors.
- What else must change: all of B15 first. The diagnosis compares top-3 feature-name sets of hit vs missed survivors in the Top-20 (`:743-764`); with no hit it returns `training_issue` unconditionally (`:758-760`). Contributions are stored as normalised absolute values (`per_survivor_attribution.py:108-115`); the system map requires signed values for a strength-seeking use (`:121-125`).

**S6 — Activating attribution piecemeal.**
- Ruling: system map `:272` ("One-line Ch13 feature-name patch alone — Not sufficient; do not land in isolation"), `:275` ("Real attribution run — Blocked until schema, scaler and prediction-parity gates pass"), gate A-NORM `:99`. The map is Alpha-authored and quotes the Beta review as binding; the review is not a separate file in this clone.
- If lifted: the blockers could be fixed one at a time, and attribution run before prediction parity is shown.
- What else must change: for tree models, Blocker 0 and Blocker 1 are sufficient to make it run (Blocker 2 already has a fallback). For the neural net, Blockers 2, 3 and 4 all remain, and without the scaler the result is computed at a different point in feature space than Step 6 predicts from.

**S7 — Selfplay may not see live outcomes or promote.**
- Ruling: `docs/CONTRACT_SELFPLAY_CHAPTER13_AUTHORITY_v1_0.md:4-6` (RATIFIED, Team Beta + User), invariants `:106-123`, forbidden actions `:188-193`.
- If lifted: selfplay could be scored on graded draws, or a candidate could be promoted without a human.
- What else must change: selfplay has no reader for graded outcomes. Auto-promotion has no caller: after a selfplay run WATCHER only logs that a candidate awaits review (`agents/watcher_dispatch.py:182-191`), and `selfplay.auto_promote` / `require_human_approval` (`watcher_policies.json:99-100`) have no reader in tracked code. Then B11, B12, B13.

**S8 — Sieve thresholds in the LLM application path.**
- Ruling: `docs/TODO_SELFPLAY_AND_LLM_AUTONOMY.md:92-103`, and after the fix `:111-131` ("a precondition, not a green light").
- State of the precondition: section 7 below.
- If lifted: `forward_threshold` / `reverse_threshold` could be proposed and applied.
- What else must change: S3 (there is no application path to add them to); the three-value audit the note requires (`:117-123`); B17.

**S9 — Advisor-chosen `search_strategy` / sampler.**
- Ruling: reserved authority, `docs/CLAUDE_CODE_INSTRUCTIONS_BOUNDED_PHASE_6.md:98`, `:121`; `docs/S172_RUN4_ROUTING_PATCH_BRIEF_ROUTE_A.md:84`.
- If lifted: the Strategy Advisor's `search_strategy` override (`parameter_advisor.py:133`) could select the Step-1 sampler.
- What else must change: the field is read into the override dict and never consumed (`agents/watcher_dispatch.py:149-158` uses two fields only); selfplay does not run Step 1.

---

## 6. DESCRIPTION-ONLY

- **D1 — Selfplay as a "learning loop".** Chapter `:1139`; contract "Learning statistical policy" (`:64`). Governed wording is "policy-conditioned evaluation harness" / "discovery front-end" (`docs/TB_UPDATE_SELFPLAY_REFRAMING_2026-07-28.md:20-40`; system map `:60`). The code-path content of that characterisation is B10, B11, B13, B14, S4.
- **D2 — "Improves continuously" vs defensive triggers.** Statement 3. Chapter `:1092-1095`.
- **D3 — "Without human intervention", "That is autonomy".** Chapter `:49-54`, against its own `:480-482`.
- **D4 — `holdout_hits` as the rolling label.** Chapter `:126`, `:140`, `:429-435`. Step 5's target is `holdout_quality` (`meta_prediction_optimizer_anti_overfit.py:612`); a governed change the chapter predates.
- **D5 — Reference tables.** §14 file names differ from the code's (`acceptance_decisions.jsonl`, `.parameter_change_history.json`, `chapter13_cycle_history.jsonl`: `chapter_13_acceptance.py:56-57`, `chapter_13_orchestrator.py:83`). §18.7 names `parameter_change_limit`; the key is `max_parameter_delta` (`watcher_policies.json:41`). §18.1 bounds (window 100-500, offset 0-50) disagree with the live bounds the fact file records (`:1156-1162`); not re-derived here.
- **D6 — "Pydantic model for proposals".** Chapter `:963`. `llm_proposal_schema.py:70-71`, `:104-105` are dataclasses.
- **D7 — Tracker status words.** "COMPLETE" for Phase 8 and for per-survivor attribution (`v3_9:65`, `:72`); "Full Autonomy Achieved" (`docs/CHAPTER_13_SECTION_19_UPDATED.md:4`). "Real SHAP attribution" (`v3_9:104`) was on synthetic artifacts (system map `:101`). "62-feature validation" (`v3_9:183`) is stale; trained count is 89 (system map `:157`).
- **D8 — Chapter file hygiene.** `CHAPTER_13_LIVE_FEEDBACK_LOOP.md` is the older text: `_v1_1.md` differs only by one added note at its `:928`. Header says 1.1.0 (`:5`), §21 says 1.2.0 (`:1135`) and sits after "END OF CHAPTER 13" (`:1129`). The §19 checklist is unchecked (`:942-984`) while `CHAPTER_13_SECTION_19_UPDATED.md` says complete.
- **D9 — "Phase 7 complete".** `v3_9:40` means WATCHER dispatch integration (Feb 2026), not S172 Phase 7 (`docs/PIPELINE_BEHAVIOUR_MODEL.md:64-70`).

---

## 7. The 27 tasks in `TODO_SELFPLAY_AND_LLM_AUTONOMY.md`

"Done" = the code exists and is reachable. Tasks that are test runs cannot be confirmed from a clone; they are marked by whether the code they exercise exists.

| Task | Status | Anchor / note |
|---|---|---|
| A1 multi-episode selfplay | code present; run unverified | `selfplay_orchestrator.py:481-545`. Episodes are identical in input (B14) |
| A2 candidate emission | code present; run unverified | `:955-1028` |
| A3 policy history archive | code present; run unverified | `:1030-1037` |
| A4 with active policy | code present; run unverified | `:498-499`; `policy_conditioned_episode.py:140-177` |
| A5 filter transform | **open — the example as written cannot filter** | The TODO's policy (`:55-65`) uses `min_score` with no `field`; an enabled filter with no field fails schema validation (`policy_transform.py:667-669`) and the loader falls back to the empty baseline (`policy_conditioned_episode.py:161-165`) |
| A6 telemetry health | code present; run unverified | writer `modules/learning_telemetry.py:97`; reader `chapter_13_diagnostics.py:82-97` |
| A7 Chapter 13 candidate validation | code present; run unverified | `chapter_13_acceptance.py:945-946`, `:967-995`. Under tracked thresholds a low-R² candidate is rejected (B13) |
| B1 proposal models | done (as dataclasses, D6) | `llm_proposal_schema.py:70-71`, `:104-105` |
| B2 GBNF grammar | done | `agent_grammars/chapter_13.gbnf`. The advisor loads the older `grammars/chapter_13.gbnf` (`chapter_13_llm_advisor.py:310-312`), which lacks the `step1_relaunch` scope |
| B3 parameter vocabulary | **open** | `chapter_13_parameter_vocabulary.py` not in the tracked tree (history and gitignored files not visible). See B17 |
| B4 diagnostics | done | `chapter_13_diagnostics.py:750-884`. Inputs: B1 |
| B5 history archiver | done | `:891-926` |
| B6 test on a real run | unverified | runtime |
| B7 prompt with diagnostics + vocabulary | partial | Prompt `chapter_13_llm_advisor.py:95-176` has no vocabulary and no frozen list; the orchestrator passes no run history (`chapter_13_orchestrator.py:386`) |
| B8 router integration | done | `chapter_13_llm_advisor.py:62`, `:299-344` |
| B9 parse to proposal | done | same block; fallback `:346-436` |
| B10 mock → LLM → proposal test | unverified | runtime |
| B11 `validate_proposal()` in WATCHER | superseded | lives in `chapter_13_acceptance.py:327`, called from the orchestrator `:395` |
| B12 `apply_parameter_changes()` | **open, and held by ruling** | no such function; S3 |
| B13 `dispatch_selfplay()` | done | `agents/watcher_dispatch.py:88-232` |
| B14 `dispatch_learning_loop()` | done | `:239-396` |
| B15 wire LLM advisor into WATCHER daemon | superseded | the advisor is hosted by the orchestrator (`chapter_13_orchestrator.py:134-136`, `:386`); the daemon polls requests and approvals (`agents/watcher_agent.py:3187-3209`) |
| B16 end-to-end test | superseded in form | no `--validate-proposal` in `agents/watcher_agent.py`; equivalent is `chapter_13_acceptance.py --validate` (`:939`) or `chapter_13_orchestrator.py --once` |
| B17 auto-accept → execution | **open** | accept ends at an approval request (`chapter_13_orchestrator.py:398-425`); S1, S3, B16 |
| B18 auto-reject (delta) | done | `chapter_13_acceptance.py:400-414` |
| B19 escalation → human alert | done | `:457-504`; request `chapter_13_orchestrator.py:427-430`; Telegram `agents/watcher_agent.py:3376-3383` |
| B20 selfplay dispatch from WATCHER | done | CLI `agents/watcher_agent.py:3549`; `agents/watcher_dispatch.py:117-123` |

Totals: done 11 · superseded 3 · open 4 (A5, B3, B12, B17) · partial 1 (B7) · code present, run unverified 6 (A1-A4, A6, A7) · unverified 2 (B6, B10).

**The "BLOCKED-BY (S172 Phase 5 D6)" note (`TODO:82-103`).** The defect it describes does not hold at this commit:
- `build_stripe_assign_payload` takes `forward_threshold` and `reverse_threshold` as required keyword arguments and emits the resolved value (`miner/range_miner_coordinator.py:9216-9263`).
- The worker no longer chooses. The `0.25` default remains only for a payload that carries no threshold field (`miner/range_miner_worker.py:1165-1172`; the note's `:734` anchor has drifted), contradictory fields are refused (`:1177-1185`), and the value the kernel ran with is reported (`:1095`, `:1294-1302`).
- The gate file named in the note exists: `tests/test_s172_phase5_d6_threshold_path.py` (not run here).
- I could not confirm the commit id `2be51d5` from the fact file §2.7: the mirror holds one commit and `git cat-file -t 2be51d5` fails (U4).
- The note's own status update (`TODO:105-131`) still stands: the fix is a precondition, and its three remaining conditions are unmet in code (S3, S8). The `parameter_application` flag it mentions is still unread.

---

## 8. Surfaces searched, and surfaces unavailable

**Searched and read**
- Governance: `docs/TB_RULING_S176/S177/S178/S179*`, `docs/PROPOSAL_WATCHER_KPI_GOVERNANCE_STATES_v1_1.md` (§7), `docs/TB_UPDATE_SELFPLAY_REFRAMING_2026-07-28.md`, `docs/CONTRACT_SELFPLAY_CHAPTER13_AUTHORITY_v1_0.md`, `docs/TEAM_ALPHA_AUTONOMY_CONTROL_SURFACE_SUBMISSION.md`, `docs/CLAUDE_CODE_INSTRUCTIONS_CHAIN_C_TRUTHFULNESS_HOTFIX.md`, `docs/TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md` (all), `docs/S172_ATTRIBUTION_AND_FEATURE_TRACE_REPORT.md` (§2.7-2.8), `docs/WATCHER_POLICIES_REFERENCE.md`, `docs/PHASE_9B2_INTEGRATION_SPEC.md`, `docs/PROJECT_FILE_CATALOG.md` (selfplay/KPI rows, §7 gap 8), the named sections of the fact file and the behaviour model.
- Chapters: `CHAPTER_13_LIVE_FEEDBACK_LOOP.md` (all, diffed against `_v1_1` and `SECTION_19_UPDATED`), `CHAPTER_13_IMPLEMENTATION_PROGRESS_v3_9.md` (all).
- Code, read in full: `chapter_13_orchestrator.py`, `chapter_13_triggers.py`, `chapter_13_acceptance.py`, `selfplay_orchestrator.py`, `policy_conditioned_episode.py`. Read in the relevant parts: `chapter_13_diagnostics.py`, `chapter_13_llm_advisor.py`, `llm_proposal_schema.py`, `per_survivor_attribution.py`, `inner_episode_trainer.py`, `policy_transform.py`, `agents/watcher_dispatch.py`, `agents/watcher_agent.py`, `prediction_generator.py`, `synthetic_draw_injector.py`, `miner/range_miner_coordinator.py`, `miner/range_miner_worker.py`, `watcher_policies.json`, `configs/selfplay_config.json`, `agent_manifests/*.json`, `agent_grammars/chapter_13.gbnf`.
- Absence claims were made by searching all tracked `*.py`, `*.sh`, `*.json` (excluding `apply_s*`, `verify_s*`, `backups/`), then reading each hit.

**Not available, and where it matters**
- **Host state** (systemd, cron, deployed uncommitted files on Zeus / VM101). Decides B1, B2 and whether S179's implementation exists outside the tracked tree.
- **Gitignored files** (`.gitignore:41` `*.json`): `agent_manifests/definitions.json`, `learned_policy_*.json`, `prediction_pool.json`, `lottery_history.json`, `pending_approval.json`, all run artifacts. Note that `watcher_policies.json` and `configs/selfplay_config.json` are tracked here; the host copies may differ.
- **Git history.** One commit. No deleted-file search, no commit verification.
- **Private repo, pre-repository archives on ser8.**
- **Documents named by the system map but not in this clone** (`:284-286`): `TODO_TFM_FEATURE_CONTRACT_AND_ATTRIBUTION_ACTIVATION_v1_0.md`, `PROPOSAL_SELFPLAY_LEARNING_LOOP_REV2_1_ADDENDUM.md`, `PROPOSAL_TFM_ATTRIBUTION_DRIVEN_DISCOVERY_LOOP_v1_0.md`. Also the standalone Beta ruling texts behind S3 and S6. Any of these could carry a ruling that changes a label above.
- **Not read:** `SELFPLAY_ARCHITECTURE_PROPOSAL_v1_0.md`, `PROPOSAL_EPISTEMIC_AUTONOMY_UNIFIED_v1_3.md`, `CONTRACT_LLM_STRATEGY_ADVISOR_v1_0.md`, the earlier progress trackers (v1_2 to v3_8), the session changelogs, the `apply_s*.py` patch corpus. The fact file warns that the patch corpus carries governance in docstrings; a B-item marked "not found in the governance trail" was not checked against it.

---

## 9. Accurate claims and unverified claims (for the count)

**Accurate (25)**
1. New-draw flag and halt-flag gating (chapter `:190-196`) — `chapter_13_orchestrator.py:158-175`, `:215-233`.
2. Synthetic injection needs both flags (`:291-304`) — `synthetic_draw_injector.py:51-77`.
3. Injector takes PRNG type from `optimal_window_config.json` (`:227`) — `:84-100`.
4. Diagnostics written and archived (`:344`, `:715-716`) — `chapter_13_diagnostics.py:891-926`.
5. Triggers are defined and evaluated; execution needs approval (`:476-482`) — `chapter_13_triggers.py:201-370`.
6. N-draws ≥ 10, drift < 0.2, misses ≥ 5 as policy thresholds (`:490-492`) — `:246-288`; `watcher_policies.json:19-23`.
7. LLM is advisory and does not execute (`:516`, `:530-536`) — `chapter_13_orchestrator.py:373-441`.
8. System prompt as printed (`:540-554`) — `chapter_13_llm_advisor.py:95-104`.
9. Reject on confidence < 0.60 (`:656`) — `chapter_13_acceptance.py:359-370`.
10. Reject on more than 3 parameters (`:657`) — `:372-383`.
11. Reject on delta > 30% (`:653`) — `:400-414`, `:570-596`.
12. Auto-accept needs low risk, confidence ≥ 0.75, ≤ 2 parameters (`:664-666`) — `:527-552`.
13. Escalate on medium/high risk, regime/chaotic flags, ≥ 3 consecutive failures (`:674-677`) — `:457-478`.
14. Action → step lists for partial and full reruns (`:768-782`) — `chapter_13_triggers.py:450-458`.
15. Human halt and approve/reject controls, LLM can be disabled (`:844-849`) — `chapter_13_orchestrator.py:1003-1019`, `:1033-1040`, `:1082-1084`.
16. Selfplay emits `learned_policy_candidate.json` (`:1148`) — `selfplay_orchestrator.py:1018-1026`.
17. `--validate-selfplay` / `--promote` exist with the stated thresholds (`:1156-1173`) — `chapter_13_acceptance.py:64-67`, `:945-948`.
18. `selfplay_health` block in diagnostics (`:1178-1189`) — `chapter_13_diagnostics.py:871-881`.
19. Only Chapter 13 writes `learned_policy_active.json` (`:1226`) — sole writer `chapter_13_acceptance.py:848`.
20. Telemetry is observational only (`:1228`) — `chapter_13_diagnostics.py:82-97`; no trigger reads it.
21. Tracker: policy transform module and policy-conditioned mode exist (`v3_9:43-44`) — `policy_transform.py:489`; `selfplay_orchestrator.py:725-755`.
22. Tracker: Phase 9B.3 is future work (`v3_9:45`) — `policy_conditioned_episode.py:398-401`.
23. Tracker: Strategy Advisor called before selfplay (`v3_9:121`) — `agents/watcher_dispatch.py:472-502`.
24. Tracker: root cause is observe-only (`v3_9:89`) — `chapter_13_orchestrator.py:321-323`.
25. Tracker: episode diagnostics and trend detection in selfplay, observe-only (`v3_9:78-82`) — `selfplay_orchestrator.py:514-524`, `:826-868`.

**UNVERIFIED (7)**
- **U1** Soak A/B/C and Phase 8 soak results (`v3_9:53-55`, `:91-108`). Would verify: the soak logs and `chapter13_cycle_history.jsonl` on the host.
- **U2** Whether the host has a producer for `prediction_pool.json` or a bridge between `lottery_history.json` and the pipeline dataset. Would verify: `ls -la` of the project root on Zeus, the systemd units, and one real `post_draw_diagnostics.json` showing `pool_size`.
- **U3** Whether the S179 implementation landed anywhere. Would verify: `grep -rn kpi_governance` on the host tree and the private repo (also catalog §7 gap 8, `docs/PROJECT_FILE_CATALOG.md:1045`).
- **U4** That commit `2be51d5` is the D6 fix. Would verify: `git show 2be51d5` in a repo with history.
- **U5** Convergence targets on synthetic data, Hit@20 > 5% (chapter `:283-287`). Would verify: a recorded synthetic run.
- **U6** Measured R² ≈ 0.000155 on real data (fact file `:344`), which B13 relies on. Would verify: one selfplay run's `learned_policy_candidate.json`.
- **U7** That Part A tasks A1-A4, A6, A7 were ever run. Would verify: `policy_history/` and `telemetry/` on the host.

---

## Addendum — live-host re-verification, 2026-11-11

**Host:** VM 101 (`zeus-ubuntu`), user `michael`, `/home/michael/distributed_prng_analysis`.
**Base:** `git rev-parse HEAD` = `a15b5247336255801de8cf67d52653701b1e8c80` (`a15b524`), the same commit the
audit above was made from. **Authority:** `docs/CLAUDE_CODE_INSTRUCTIONS_CHAPTER_CORRECTIONS_20261111.md` REV1, §3.1
(re-verify, do not redo) and §5 (Chapter 13 host question). The text above is left exactly as written; this
addendum only adds to it. Read-only throughout: no service was started, stopped or edited; nothing was run except
reads, `systemctl`/`journalctl` listings, `git log`/`git show`, and small `python3 -c` tallies over existing JSON files.

### A.0 Null results first (each with its denominator)

1. **No host-side bridge writes `prediction_pool.json`.** Writers found: **0**, across 17 repo files that mention the
   name (tracked + untracked + gitignored text files ≤ 2 MB), 1 file in `~/` outside both projects, 0 in `/usr/local/bin`,
   0 systemd units, 0 cron entries, and **0 commits in `git log --all`** that ever put the string into
   `prediction_pool.json`'s supposed producer, `prediction_generator.py`. The file does not exist on the host.
2. **Nothing on the host raises `new_draw.flag` except the two tracked test/manual tools** (`synthetic_draw_injector.py`,
   `draw_ingestion_daemon.py`). No unit, timer, cron job or running process does. The flag file does not exist now.
3. **Every diagnostics cycle the host ever recorded graded an empty pool: 233 of 233** archived files in
   `diagnostics_history/` have `prediction_validation.pool_size == 0`. **All 233 were synthetic draws** (230
   `synthetic`, 3 `synthetic_bootstrap`); **0 of 233 were real draws.** Dates: 3 in 2026-01, 230 in 2026-02. Nothing since.
4. **The attribution gate never opened on host data: 0 of 233** diagnostics carry a flag containing both "hit_rate"
   and "drop", and **0 of 233** carry a `hit_at_20` key (B15 Blocker 0, now confirmed on real artifacts).
5. **The cooldown was never reset by an executed retrain:** `.chapter13_cooldown_state.json` has `last_retrain_at: null`,
   `runs_since_retrain: 236`, although `watcher_requests/archive/` holds **39** `learning_loop` requests marked
   `completed_by_watcher` and **1** `completed_by_cli` (B7, confirmed on real artifacts).

### A.1 Host surfaces read

| Surface | Command | Result |
|---|---|---|
| system units | `systemctl list-units --all`, `systemctl list-unit-files`, `ls -la /etc/systemd/system /etc/systemd/system/multi-user.target.wants` | One project unit: `daily3scraper.service` — **disabled**, `inactive (dead)`. `ExecStart=/usr/bin/python3 /home/michael/distributed_prng_analysis/run_daily3scraper.py`, `Restart=always`, `User=michael`. Other custom units (`cluster-boot-notify`, `netconsole-listener`, `nvidia-*`) have nothing to do with draws or pools |
| unit journal | `journalctl -u daily3scraper.service` | Every start fails: `can't open file '.../run_daily3scraper.py': [Errno 2] No such file or directory` (first lines 2026-05-14 19:40; last attempt 2026-07-29 15:43, "Start request repeated too quickly") |
| runner file | `ls run_daily3scraper.py`; `git log --all -- '*run_daily3scraper*'`; `find / -name 'run_daily3scraper*'` (excluding `~/piece_matcher_project`) | Absent from the working tree, from **all** git history, and from the whole filesystem |
| scraper itself | `/bin/grep -nE 'flag\|lottery_history\|daily3\.json\|write' daily3_scraper.py` | Writes `daily3.json` only (`daily3_scraper.py:14`, `:99`). No flag, no `lottery_history.json`. `~/daily3_scraper.py` is byte-identical (`diff -q`) |
| user units | `systemctl --user list-units --all --type=service,timer`; `ls ~/.config/systemd/user` | `sunshine.service` only. Nothing project-related |
| timers | `systemctl list-timers --all` | 16 timers, all OS housekeeping |
| cron | `crontab -l` (→ "no crontab for michael"); `ls -la /etc/cron*`; `/etc/crontab` | OS entries only. `/var/spool/cron/crontabs` → **UNAVAILABLE** (permission denied; other users' crontabs not readable as `michael`) |
| processes | `ps -eo pid,user,lstart,args`, filtered for `chapter_13\|draw_ingestion\|synthetic_draw\|watcher_agent\|daily3` | None running |
| untracked + ignored files | `git status --ignored --porcelain` (457 entries), `git ls-files -o -i --exclude-standard` (81,552 files); `find … -size -2M` over `*.py *.sh *.service *.timer *.json *.yaml *.txt *.cfg *.ini *.toml Makefile` then `/bin/grep -lI 'prediction_pool\.json\|new_draw\.flag'` | 17 hits, listed in A.2 |
| home outside the projects | same search, `/home/michael` depth ≤ 3, pruning `~/piece_matcher_project` (not read), `~/distributed_prng_analysis`, caches, venvs | 1 hit: `~/s167_safety_backup_20260424_170757/watcher_agent.py:1584`, a backup copy of the WATCHER step-output table (reader-side declaration, not a writer). `~/bin` is empty |
| `/usr/local/bin` | `ls`; `/bin/grep -l 'prediction_pool\|new_draw'` | 2 notify scripts, no hit |
| git history | `git log --all -S 'prediction_pool.json'` (25 commits), `-S 'new_draw.flag'` (21 commits), `-S 'confidence_map.json'`, `-- prediction_generator.py`, `git show c6e3989`, `git show eb44bec` | See A.3 |
| runtime artifacts | `post_draw_diagnostics.json`, `pending_approval.json`, `.chapter13_cooldown_state.json`, `chapter13_cycle_history.jsonl` (231 lines), `diagnostics_history/` (233), `watcher_requests/archive/` (61), `acceptance_decisions.jsonl`, `learned_policy_active.json`, `learned_policy_candidate.json`, `policy_history/` (51), `telemetry/` (2), `llm_proposals/` (224), `lottery_history.json`, `daily3.json`, `predictions/next_draw_prediction.json`, `predictions/history/` (19) | See A.4 |
| `agent_manifests/definitions.json` | `cat -n` (gitignored, `.gitignore:41 *.json`) | See A.5 |

**Not read (out of scope or unreachable):** `.127` bare-metal fallback (Zeus runs one OS at a time; not booted) —
**UNAVAILABLE**; rig CT100s (no rig access in this pass) — **UNAVAILABLE**; the private remote and ser8 archives —
**UNAVAILABLE**; root's and other users' crontabs — **UNAVAILABLE**; files > 2 MB were excluded from the text search
(data files; a writer script would not be that large); `~/piece_matcher_project` deliberately not touched (separate project).

### A.2 The 17 repo hits, by role

Readers or declarations of `prediction_pool.json`: `chapter_13_diagnostics.py:16`, `:19`, `:50`, `:987`;
`agents/watcher_agent.py:2223`; `watcher_agent.py:1322` (root-level copy); `agents/pipeline/pipeline_step_context.py:148`;
`schemas/results_schema_v1.json:928`; `bundle_factory.py:134` and `agents/contexts/bundle_factory.py:134` (prompt
text); `backups/phase1_patch_20260127_194227/watcher_agent.py:1172`; three **gitignored, superseded** Step 6 manifests —
`step6_restoration/agent_manifests/prediction.json:14`, `:36`; `backups/phase1_patch_20260127_194227/prediction.json:14`;
`backups/step6_restoration_20251223_213936/agent_manifests/prediction.json:14`, `:37`. **No writer.**

Raisers of `new_draw.flag`: `synthetic_draw_injector.py:39`, `:349`, `:408` and `draw_ingestion_daemon.py:48`, `:265`
(both tracked). Readers / clearers: `chapter_13_orchestrator.py:73`, `:215-239`; `draw_ingestion_daemon.py:287`, `:300`;
`run_soak_c.sh:184`, `:197`, `:306` (cleanup); `watcher_policies.json:80` (comment); `patch_operating_guide.py:148` (doc text).

### A.3 Git history (newly available) — what it adds

- `eb44bec` (2026-01-11) *"fix(manifest): Step 6 success_condition matches actual output path"* removed
  `prediction_pool.json` from `agent_manifests/prediction.json`, and the same commit records WATCHER escalating because
  `Missing required files: ['prediction_pool.json']`.
- `a7c13f7` (2026-01-12) *"feat: Chapter 13 Live Feedback Loop - Phases 1-6 complete"* introduced the Chapter 13 code
  that defaults to `prediction_pool.json` — **one day after** the manifest stopped declaring it.
- `c6e3989` (2026-01-28) *"fix: Step 6 canonical output path (Team Beta ruling)"*: "Save to
  `predictions/next_draw_prediction.json` (WATCHER contract) … Per Team Beta ruling: single canonical output".
- `git log --all -S 'prediction_pool.json' -- prediction_generator.py` returns **nothing**: Step 6 never wrote that file
  in any commit. Before `c6e3989` it wrote `predictions/predictions_<timestamp>.json`.
- `git log --all -S kpi_governance` → only `833507e` (docs + six `watcher_kpi_*.py` / `step0_heuristic_validation.py`
  analyzers) and `131787d` (dataset pointer). `git log --all -S audit_only -- '*.py'` → nothing.
- `git log --all -- '*chapter_13_parameter_vocabulary*'` → nothing; the file has never been committed and is not on disk.
- `git show -s 2be51d5` → `2be51d5 2026-07-29 S172 Phase5 D6 B: directional threshold propagation to kernel,
  effective-threshold provenance + parent-side fail-closed enforcement …`.
- `git log -S 'request_selfplay(' -- chapter_13_triggers.py chapter_13_orchestrator.py` → only `358e615` (the commit that
  defined it). No commit ever added a caller.

### A.4 Host runtime artifacts — what they show (all point-in-time, last written Feb–Mar 2026)

- `post_draw_diagnostics.json` (2026-02-15, draw `SYNTHETIC-2026-02-15-424`): `pool_size 0`, `exact_hits 0`,
  `pool_coverage 0.0`, `predicted_vs_actual_correlation 0.0`, `consecutive_misses 233`, `decay_candidates []`,
  `reinforce_candidates []`, `survivor_churn 0.0`, `window_decay 0.0`. Matches B1 and B8 exactly.
- `pending_approval.json`: trigger `hit_rate_collapse`, reasoning "Hit rate (0.0000) < collapse threshold (0.01)",
  `steps_to_run [3, 5, 6]`, approval text recommends `python3 chapter_13_triggers.py --approve`; `status: "failed"`,
  `approved_at …15:55:46.108031`, `completed_at …15:55:46.108372` — approved and failed 0.34 ms apart, which is the B5 path
  (`approve_request` → `execute_learning_loop` returns False at the `watcher_agent is None` check).
- `chapter13_cycle_history.jsonl` (231 parseable cycles): outcomes `auto_executed_test_mode` 120, `pending_approval` 77,
  `proposal_rejected` 29, `escalated` 5. The 120 are the B16 label (no execution follows it in code).
- `watcher_requests/archive/` (61 files): `learning_loop / completed_by_watcher` 39, `LEARNING_LOOP / completed_by_cli` 1,
  `selfplay_retrain` 21 (20 written by hand-named test sources — `soak_test_B`, `phase7_d5_*`, `try1_*`, `chapter_13_test` —
  and one with `requested_by: chapter_13_triggers`, 2026-01-30, i.e. `request_selfplay` was invoked once outside the code
  paths; it still has no caller in tracked code).
- `acceptance_decisions.jsonl`: `accept` 195, `reject` 29, `escalate` 5; 5 entries of type `selfplay_candidate`
  (the A7 validation path was exercised, first 2026-01-30).
- `learned_policy_active.json` (promoted 2026-01-30 by `chapter_13_acceptance`): `val_r2 0.99999`, `fitness 0.847`, and
  **no `transforms` key** (consistent with B11).
- `learned_policy_candidate.json` (2026-02-24): `val_r2 −0.00029`, `fitness −0.0014`, `transforms {}`, has
  `parent_policy_id` and `fingerprint` (which `SelfplayCandidate.from_dict` would discard, B11).
- `lottery_history.json`: 5,148 draws — `synthetic_bootstrap` 5,000, `synthetic` 148; last updated 2026-02-15. `daily3.json`:
  18,068 entries, last `2026-02-26 midday`. **The two histories share no writer** (B2 on host data).
- `predictions/next_draw_prediction.json` (2026-03-07): `predictions` is a list of two ints `[436, 429]`,
  `metadata.pool_size 20`, `intersection_count 2`, `model_type neural_net`, `feature_count 89`. This is what Step 6
  writes; Chapter 13 never reads it.
- **`optimal_window_config.json` is absent from the project root now** (only `optimal_window_config_test.json`,
  `optimal_window_config.s162_victory.json` and copies under `logs/`, `old_results/`). `synthetic_draw_injector.py:93-97`
  raises `FileNotFoundError` in that case, so the injector would refuse to run today. Host state, not a code defect.

### A.5 `agent_manifests/definitions.json` (gitignored; 220 lines, schema 1.1.0, `updated_at` 2026-01-02)

Declares Step 6 outputs as `predictions/next_draw_prediction.json`, `predictions/prediction_pools.json` and
`predictions/ranked_survivors.json` (`definitions.json:80-84`). **None of the three is `prediction_pool.json`** or
`predictions/ranked_predictions.json` (what `chapter_13_orchestrator.py:877` reads). On the host only the first exists.
`prediction_pools.json` is the default output of `build_pools.py:174` (not part of Step 6). No tracked `*.py`/`*.sh`
reads `definitions.json` (`/bin/grep -l definitions\.json` over tracked code → no file). Its sidecar example still
names `holdout_hits` as the target and `feature_count 62` (`:178`, `:185`). It adds a fourth naming of the Step 6 output
to the B1 disagreement and a third name to the B15 Blocker 1 artifact; it changes no label.

### A.6 Item-by-item result

**Anchor re-verification.** `git diff --stat HEAD` is empty: every tracked file on the host is byte-identical to
`a15b524`. About 200 anchors cited above were opened by line on the host (`sed -n`), covering every code file the audit
cites; **all matched** the cited symbol or statement. `watcher_policies.json` and `configs/selfplay_config.json` on the
host are the tracked copies, unchanged.

**Confirmed (61): labels and anchors stand as written.** Accurate 1–25; B1–B18; S1–S9; D1–D9. Host artifacts add
direct runtime confirmation for B1 (233/233 empty pools), B2 (disjoint histories), B4 (`HIGH_WINDOW_DECAY` appears in
1/233, consistent with a 0-or-1 metric), B5 (A.4 pending approval), B7 (A.0 #5), B8 (empty reinforce/decay lists),
B10 (no code caller; one hand-invoked request), B11 (active policy has no `transforms`), B15 Blocker 0 (0/233), B16 (120
labelled cycles).

**Corrected (3) — supporting text, not labels:**
- **C1, B1 "not found in the governance trail."** Partly wrong once history is visible: the **Step 6 side** of the
  mismatch is governed — `c6e3989` cites a Team Beta ruling for `predictions/next_draw_prediction.json` as the single
  canonical output, and `eb44bec` removed `prediction_pool.json` from the manifest the day before Chapter 13's code
  landed. No governance document says Chapter 13 should keep reading a file nobody writes, so **B1 stays BROKEN**.
  The standalone ruling text behind `c6e3989` was not found in `docs/TB_*` (searched for "canonical output" /
  "single canonical"); only the commit message records it.
- **C2, B2 "the fact file says an enabled boot-triggered scraper unit exists on the host."** On VM 101 the unit
  exists but is **disabled**, and its `ExecStart` target `run_daily3scraper.py` has never existed in git history or on
  disk; every recorded start failed (A.1). The fact file's statement (`docs/TFM_PROJECT_FACTS_SKILL.md:200-202`) is
  therefore **stale for VM 101**; it may describe `.127`, which is UNAVAILABLE. Even if it ran, `daily3_scraper.py` writes
  `daily3.json` only and raises no flag. **B2 stays BROKEN.**
- **C3, §8 "the host copies [of `watcher_policies.json`, `configs/selfplay_config.json`] may differ."** They do not
  (tracked, `git diff` clean).

**Added (1), no label change:** B15 Blocker 1 — `definitions.json:83` names a third, also-unproduced artifact
(`predictions/ranked_survivors.json`) for the ranked output (A.5).

**UNVERIFIED items:**
- **U2 — RESOLVED: no bridge.** A.0 #1–#2, A.1. B1 and B2 are breaks on the host, not artefacts of reading a clone.
- **U3 — RESOLVED: the S179 implementation did not land** — not in the tracked tree, not in any commit
  (`-S kpi_governance` touches only docs/analyzers; `-S audit_only -- '*.py'` empty), not in any of the 31 untracked
  `*.py`/`*.sh` on the host (`/bin/grep -lE 'kpi_governance|BLOCKED_BY_GOVERNANCE|manual-governance-override|unique_rank'` → none).
- **U4 — RESOLVED:** `2be51d5` exists and is the S172 Phase 5 D6 B threshold-propagation commit (2026-07-29).
- **U6 — RESOLVED in substance:** the host's last candidate has `val_r2 −0.00029`, `fitness −0.0014`, and the last
  diagnostics `fitness_best −0.0004`. The exact 0.000155 figure was not re-measured, but what B13 relies on (R² far below
  0.05, so fitness ≤ 0) holds on host data. Note one earlier promotion **did** pass (`learned_policy_active.json`,
  2026-01-30, `val_r2 0.99999`); B13's "cannot pass" is about the present objective on present data, not a universal claim.
- **U7 — RESOLVED:** `policy_history/` holds 51 policy files (from 2026-02-03), `telemetry/` holds
  `learning_health.jsonl` + `learning_health_latest.json`, `acceptance_decisions.jsonl` holds 5 `selfplay_candidate`
  validations: A1–A4, A6 and A7 were run.
- **U1 — PARTLY RESOLVED:** cycle history and request archive tallies are in A.4. The soak logs themselves were not
  opened, so "Soak A/B/C PASSED" remains as the tracker states it. The tracker's "73 auto-executed" is the B16 label;
  the host history has 120 such cycles and none of them executed through that route.
- **U5 — still UNVERIFIED:** no host record measures Hit@20 against the chapter's > 5 % target, and with 233/233 empty
  pools Chapter 13 itself never could.

**Tally for this addendum:** confirmed **61**, corrected **3** (supporting text; labels unchanged), added **1**,
UNVERIFIED resolved **5** (U2, U3, U4, U6, U7) + **1 partly** (U1), still UNVERIFIED **1** (U5) + the unresolved half of U1.

### A.7 Chapter 13 §5 host question — answer

> *Is there a host-side bridge that writes `prediction_pool.json` or raises `new_draw.flag`?*

**No.** On VM 101 at `a15b524` nothing writes `prediction_pool.json` (no file, no unit, no cron job, no untracked or
ignored script, no commit in history), and nothing raises `new_draw.flag` except the two tracked test/manual tools. The
only project systemd unit, `daily3scraper.service`, is disabled, points at a runner that does not exist, and would write
`daily3.json` only. The loop's runtime record (233 diagnostics, all synthetic, all with an empty pool) is consistent with
that. `.127`, the rigs and root/other-user crontabs are **UNAVAILABLE** and are not reported as clean.

### A.8 Files opened for this addendum (execution proof)

`docs/CLAUDE_CODE_INSTRUCTIONS_CHAPTER_CORRECTIONS_20261111.md`, `docs/CHAPTER_13_AUDIT.md`,
`docs/CHAPTER_13_LIVE_FEEDBACK_LOOP.md`, `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` (header, headings, §13–§14; not edited),
`docs/PIPELINE_BEHAVIOUR_MODEL.md` §1.1, `docs/TODO_CONSOLIDATED.md` (Chapter 13 rows), `docs/TFM_PROJECT_FACTS_SKILL.md:198-203`,
`docs/TB_RULING_S177_KPI_GOVERNANCE.md:300-328`, `docs/CHAPTER_13_IMPLEMENTATION_PROGRESS_v3_9.md:50-56`,
`docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md:281-305`, `docs/S172_WINDOW_ANCHOR_BRIEF_I.md:360-368`;
code by line: `chapter_13_orchestrator.py`, `chapter_13_triggers.py`, `chapter_13_acceptance.py`, `chapter_13_diagnostics.py`,
`agents/watcher_agent.py`, `agents/watcher_dispatch.py`, `selfplay_orchestrator.py`, `inner_episode_trainer.py`,
`policy_conditioned_episode.py`, `policy_transform.py`, `prediction_generator.py`, `synthetic_draw_injector.py`,
`draw_ingestion_daemon.py`, `per_survivor_attribution.py`, `meta_prediction_optimizer_anti_overfit.py`, `window_optimizer.py`,
`daily3_scraper.py`, `watcher_policies.json`, `configs/selfplay_config.json`, `agent_manifests/window_optimizer.json:212-218`,
`agent_manifests/definitions.json` (all); host: `/etc/systemd/system/daily3scraper.service`,
`/etc/systemd/system/cluster-boot-notify.service`, and the runtime artifacts listed in A.1.

### A.9 One new finding from the chapter-correction pass (not in §1–§9 above)

**B19 — Feature diagnostics read a sidecar path and keys Step 5 does not write.** *(BROKEN; not found in the
governance trail — `/bin/grep -rln 'load_model_meta|compute_feature_diagnostics|schema_hash_match' docs/` hits only the
two Chapter 13 text copies.)*
- Chapter `:332` (inputs table, `best_model.meta.json` from Step 5) and §8.2 `feature_diagnostics`.
- Code: default path is the project-root `best_model.meta.json` (`chapter_13_diagnostics.py:52`), and the orchestrator
  passes defaults (`chapter_13_orchestrator.py:282`). Step 5's sidecar is `models/reinforcement/best_model.meta.json`
  (`agent_manifests/definitions.json:12`). A missing file silently becomes `{"model_type": "unknown", "feature_hash": None}`
  (`chapter_13_diagnostics.py:184-190`). Even with the right file, the reader takes top-level `feature_hash` /
  `schema_hash` / `feature_importance` (`:448-449`); the host sidecar's top-level keys are `agent_metadata,
  checkpoint_format, checkpoint_path, data_context, feature_schema, hardware, hyperparameters, model_type, optuna,
  provenance, schema_version, signal_quality, training_info, training_metrics`, with the hash nested as
  `feature_schema.feature_schema_hash`. So `current_hash` is always None, `schema_match` stays at its default `True`
  (`:455`, `:463-464`), and turnover/entropy stay 0.
- Host: root `best_model.meta.json` does not exist; the last diagnostics file has `current_hash null`,
  `top_features {}`, `schema_hash_match true`.
- Stops at: `chapter_13_diagnostics.py:52`.
- Repair would have to: read Step 5's sidecar at its real path and key layout, and report "unknown" rather than
  "match" when no hash is available.

Updated count with B19: BROKEN 19 (18 + 1).

**Anchor precision note.** B6 "Stops at: `agents/watcher_agent.py:3773`" is the `# Dispatch` comment; the call
`watcher.run_pipeline(min(steps), max(steps))` is `:3774`, and the daemon path computes `min`/`max` at
`:3338-3339` and calls at `:3341`. Same location, one line off; the chapter uses `:3774`.

### A.10 Count reconciliation (added 2026-11-11, after commit `741822b`)

**The batch-1 report summed to 63 for this chapter. The real total is 69, not 68, and the six missing items are
the UNVERIFIED items this addendum resolved without assigning them a final label.** The original §1 verdict
(25 + 18 + 9 + 9 + 7 = 68) is correct for the audit as first written. The addendum then:

- added **B19**, a genuinely new claim found during the chapter correction (68 → **69**);
- resolved U2, U3, U4, U6, U7 and part of U1 (A.6), but the report counted only U5 as still UNVERIFIED and did not move
  the resolved items into a bucket.

Final placement of the seven original UNVERIFIED items:

| Item | Resolution (A.6) | Final placement |
|---|---|---|
| U1 soak results | partly resolved; soak logs not opened | **UNVERIFIED** (stays) |
| U2 host bridge for `prediction_pool.json` | no bridge exists (A.7) | **absorbed into B1/B2**: host confirmation of those breaks, not a separate claim |
| U3 S179 implementation landed? | did not land | **absorbed into S2** (KPI triggers held audit-only under S176/S179) and B3's "governed" status; not a separate claim |
| U4 `2be51d5` is the D6 fix | confirmed | **Accurate** |
| U5 Hit@20 > 5 % on synthetic data | no record | **UNVERIFIED** (stays) |
| U6 R² ≈ 0.000155 on real data | resolved in substance (host `val_r2` −0.00029); the exact figure was not re-measured | **Accurate (in substance)**. The figure itself is unconfirmed; B13 relies only on R² ≪ 0.05, which holds |
| U7 Part A tasks A1–A4, A6, A7 were run | confirmed by host artifacts | **Accurate** |

**Final counts for Chapter 13 (supersede the batch-1 report's 25 / 19 / 9 / 9 / 1):**

```
Accurate                  28   (25 + U4, U6, U7)
BROKEN                    19   (B1–B18 + B19)
SWITCHED-OFF-BY-RULING     9
DESCRIPTION-ONLY           9
UNVERIFIED                 2   (U1, U5)
Absorbed into an item      2   (U2 → B1/B2; U3 → S2/B3), not counted separately
-------------------------------
Total                     69   (68 audited + B19)
```

The A.6 line "still UNVERIFIED 1 (U5) + the unresolved half of U1" is the same fact as UNVERIFIED **2** above. U1 is
counted whole because its claim (soak A/B/C passed) is still unchecked.
