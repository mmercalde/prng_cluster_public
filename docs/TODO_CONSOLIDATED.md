# TFM consolidated to-do list

State of the public clone `/tmp/repos/prng_cluster_public` at commit `a15b524` (2026-08-28). Compiled 2026-11-11, read-only. The clone has one commit, so git history was not available as a surface.

TFM is functional mimicry of PRNG output. Nothing here concerns seed recovery or state reconstruction.

**Where the project stands at this commit.** RANGE-MINER passed Gate 12 (attempt 9, 2026-08-17). Window-Anchor Brief I is implemented and repaired but not accepted: the first production-shaped run on 2026-08-22 failed at Phase-5 assembly, the repair is committed, and the acceptance run ("Run 4") has not happened. A third run that day coincided with a host freeze that is still unexplained. The Run-4 routing patch was approved and committed on 2026-08-28. The rig fleet was powered off at the last session. The Phase-7 soak has not run. PWC and ZMQ are non-certifying, and Brief I closed the legacy fused reverse engine's dispatch routes.

**Three statements in the fact file (v27.1) and catalog (2026-08-20) are stale against later documents in the same repo**, and this report follows the later documents: Brief I exists; `G-PROD-SHAPE` has been run (and failed, partly for reasons of its own scoping); both `_bp` falsifier fields have been observed in production.

**Reading the tables.**
- Counts are of list entries. A row that stands for several entries says so ("4 entries") and is counted that many times. Sub-tasks that a source treats as one phase (for example Chapter 14 tasks 4.1–4.8) count once.
- **Waits on.** FLEET: needs the GPU rigs running. OWNER: needs the owner's decision, hands, host access or data that is not in the clone. REVIEW: the source says a ruling, brief or authorization is needed before work may begin, or the change touches a certified surface. NONE: work can begin from the repo alone; landing it still goes through the project's normal review. Where two apply, the first named is the one counted.
- Abbreviations: FF = `docs/TFM_PROJECT_FACTS_SKILL.md`; BL = `docs/BACKLOG.md`; CAT = `docs/PROJECT_FILE_CATALOG.md`; SM = `docs/TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md`; TM = the `TODO_MASTER_S*` series (first→last file the entry appears in); CL0822A / CL0822B / CL0828 = the session changelogs of 2026-08-22 (Window-Anchor Brief I; Brief I production-shape repair) and 2026-08-28; STEP23 report = `docs/S172_WINDOW_ANCHOR_BRIEF_I_CLOSURE_STEP23_REPORT.md`. Paths without a directory are in `docs/` or the repo root. Line numbers are at `a15b524`.

## 1. Summary

**300 distinct entries.**

| Status | Entries |
|---|---:|
| DONE | 65 |
| SUPERSEDED | 71 |
| OPEN | 144 |
| UNKNOWN | 20 |

**OPEN by group and by what it waits on**

| Group | FLEET | OWNER | REVIEW | NONE | Total |
|---|---:|---:|---:|---:|---:|
| (a) Needed to finish RANGE-MINER / S172 Phase 7 | 8 | 1 | 15 | 1 | 25 |
| (b) Self-learning: selfplay, Chapter 13, LLM autonomy | 4 | 1 | 16 | 6 | 27 |
| (c) Documentation and chapters | 1 | 4 | 1 | 9 | 15 |
| (d) Downstream pipeline, Steps 2 to 6 | 2 | 0 | 6 | 8 | 16 |
| (e) Infrastructure and hygiene | 0 | 5 | 13 | 0 | 18 |
| (f) Other | 3 | 13 | 23 | 4 | 43 |
| **Total** | **18** | **24** | **74** | **28** | **144** |

Two things the counts hide. First, most of group (a) is a chain: fleet power-on and its checks → Run 4 → Brief I acceptance → Brief II; the soak is a separate run on the same fleet. Second, most REVIEW entries are not waiting on a submitted request. They are items the sources record as "not scoped" or "no fix authorized", so the next step for each is a brief or ruling request.

## 2. Can start today

Every OPEN entry tagged NONE: 27 rows, 28 entries. The letter is the group. "Start" means begin the work; several of these end in a submission for review, and P0-C's final step needs real data.

- [a] `chapter1_p0` 8/12→10/12 improved with cause not established; flagged for its own investigation — CL0822A AC7 section
- [b] Bundle Factory Track 2: three retrieval functions return `[]` — TM S101→S148; TODO_PHASE7 v3 :267, :327; Ch13 progress v3_9 :141
- [b] Feed `results/` trial history to the Strategy Advisor — TM S101→S127
- [b] Attribution activation P0-C: ranked-prediction artifact (no writer exists), canonical model loader in Ch13, NN scaler, A-NORM parity gate, signed contributions, one observe-only real-data cycle — found: SM §13 P0-C, §4.2
- [b] Chapter 14 Phase 4: dashboard charts (4.1–4.8) — TM S101→S145 (as "web dashboard refactor"); Ch14 checklist
- [b] Chapter 14 Phase 5: TensorBoard (optional) — TM S132; Ch14 checklist
- [b] `_record_training_incident()` is defined and never called — TM S101→S127
- [c] Audit the eleven never-audited chapters (4–14, ~11,000 lines) — BL §1; CAT §7 gap 6; FF §8
- [c] Skill revision owed: v27.1 lacks EXEC-PIN-1, the proposed SR-3 (privileged seams), the stale-root-watcher fact, and still says Brief I does not exist and G-PROD-SHAPE never ran; third copy (ser8) unverified — BL §17, §23 skill note; CL0822A; CL0828 §4
- [c] `TODO_TFM_FEATURE_CONTRACT_AND_ATTRIBUTION_ACTIVATION_v1_0.md`, `PROPOSAL_SELFPLAY_LEARNING_LOOP_REV2_1_ADDENDUM.md`, `PROPOSAL_TFM_ATTRIBUTION_DRIVEN_DISCOVERY_LOOP_v1_0.md` — named as the document set, none present — found: SM §15
- [c] No committed document describes what `run_step3_full_scoring.sh` does (`STEP_SCRIPTS[3]` vs `full_scoring.json` divergence) — CAT §7 gap 2
- [c] Step 0 (TRSE) has no chapter — CAT §7 gap 3
- [c] Three binaries in `docs/` unread (two PDFs, `TEAM_BETA_REVIEW_kfolds_S100.docx`); the k-folds review may hold a ruling — CAT §7 gap 7
- [c] Chapter doc sync (Ch12, Ch13, Ch14, operating guide, README) by patch script — TM S120→S145; Ch13 progress v3_9 :137
- [c] `THRESHOLD_GOVERNANCE.md` still gives a 1K–10K survivor target — TM S145→S148
- [c] Chapter 2 VIR-6 residual: bodies of the 40 non-java_lcg kernels unread — Ch2 §14.3
- [d] P0-A: canonical 91-feature extraction and 89-feature model-input manifests; remove duplicated static feature lists — found: SM §13 P0-A, §9
- [d] P0-B: Step-3 semantic parity — exact metadata guard, identical batch and sequential merges, F-PARITY tests, fail closed — found: SM §13 P0-B, §7.3
- [d] Step-3 output validation floor is `< 46` against a 91/89 contract — BL §15
- [d] `full_scoring.json` declares 26 GPUs; frozen set is 25 (documentation drift) — BL §16
- [d] `java_lcg_cpu` non-zero-skip reachability audit (read-only; no fix authorized) — BL §8.1; Ch2 §12.1; FF §8
- [d] Skip-output work: first settle whether a hit mask reaches the host; then restore `skip_mean/std/entropy` behind a validity mask under a schema-version decision for the five dead placeholders — BL §2; SM §8.3
- [d] DEP-ABI-V2-NPZ-SEMANTICS: audit every consumer of canonical `offset` before nonzero generator phase is allowed — BL DEP-ABI-V2
- [d] sklearn feature-name warnings in Step 5 — TM S120→S170
- [f] `run_with_config` writes `[]` and prints success — Ch1 §17.3
- [f] TRSE `except` handler calls `logger`, not defined at module level — Ch1 §17.3
- [f] Threshold optimization heatmap (Display 2.1) and other post-run plots (2.2) (2 entries) — TODO_DISPLAY 2.1, 2.2

## 3. OPEN items

### (a) Needed to finish RANGE-MINER / S172 Phase 7

| Item | Sources (first → last) | Waits on | Blocks | Anchor |
|---|---|---|---|---|
| **Run 4** — Brief I acceptance run: pinned geometry (k=12, anchor 25, midday, 0.71/0.47, skip 6–99), Step 1 only, one trial, must reach Phase-5 assembly and publication | TB_RULING_BRIEF_I_PATCH_REVIEW (closing para) → CL0828 §7 | FLEET (owner launches) | Brief I acceptance; B5/B6/B8 and B7 classification; Brief II; VM101 memory change (BL §20) | TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md dispositions table; CL0828 §0, §7 ("Run 4 — Michael-authorized separately") |
| Pre-Run-4 infrastructure check: actual GPU power caps and off-host power telemetry (R2 post-commit step 3) | TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED → CL0828 §7 | FLEET | Run 4 | R2 ruling "After commit" step 3; CL0828 §5 step 3 PARTIAL, §7 "Unverifiable fleet-down" |
| Re-arm netconsole on power-on and confirm three NCPROOF lines before any run; correct RUNBOOK_NETCONSOLE_REARM §2 (`[RECONSTRUCTED]`) on first live execution; reboot persistence is unconfigured | LEADS L-3 follow-up 1; CL0828 §7 | FLEET | Run 4 (R2 step 3 names persistent netconsole) | LEADS.md:269-271; CL0828 §7 rows 2–3 |
| Physical circuit check: circuit map for the three rigs and per-rig supply logging across a phase boundary (forensic hypothesis H2, shared-circuit sag at the 24-GPU phase-4 resume, still co-leading) | CL0828 §7 | OWNER | closing H2; confidence in any full-fleet run | CL0828 §7 "Physical circuit check — OWED" |
| L-3: cause of the 2026-08-22 mid-run freeze (bounded to 18:42–18:52, no hypothesis) | LEADS L-3; RIG_CRASH_FORENSIC_20260822.md | FLEET | nothing ruled; lead may not be closed | LEADS.md:179-182, :272-274 |
| Brief I final acceptance: report the Run-4 outcome against the repair commit; until then F-4 / C-2 / D-3 (`offset` fusion) read "repair implemented, acceptance pending" | CL0822A "Post-commit closure still owed"; Ch1 §17.3 rows D-3, C-2; Ch2 §12 F-4, §14.3 | REVIEW (after Run 4) | Brief II; unqualified REPAIRED wording in Ch1/Ch2; D.1 differential | TB_RULING_BRIEF_I_PRODUCTION_SHAPE_FAILURE.md header ("Brief I NOT ACCEPTED"); Ch2 §14.3 F-4 row |
| B7 `remote_delete_status`: a writer exists and has never fired (0/5632, 0/29,082); B5/B6/B8 to be reclassified after a publishing run | STEP23 report §4.2; CL0822B table | FLEET (then REVIEW) | G-PROD-SHAPE PASS | S172_WINDOW_ANCHOR_BRIEF_I_CLOSURE_STEP23_REPORT.md:121-135; TB_RULING_BRIEF_I_PRODUCTION_SHAPE_FAILURE dispositions B5/B6/B8, B7 |
| `G-PROD-SHAPE` is not run-scoped: sections B and C aggregate the shared ledger and staging dir (seven aborted runs, 512 C2 orphan files), so it cannot PASS for any run on this host. Recorded, not repaired | STEP23 report §4.1 | REVIEW | Phase-7 certification by its own verifier | S172_WINDOW_ANCHOR_BRIEF_I_CLOSURE_STEP23_REPORT.md:119-131; classification report :330 (raw 15 pass / 6 fail) |
| `OBSERVABILITY_GAP_1`: `commit_trial` captures the Phase-5 exception to an in-memory dict; no log line, ledger keeps three-state status only | CL0822B table | REVIEW | next major certification sequence (Beta: repair before it; not a prerequisite to Run 4) | TB_RULING_BRIEF_I_PRODUCTION_SHAPE_FAILURE.md dispositions row OBSERVABILITY_GAP_1 |
| `HARNESS-LEDGER-ORDER-1`: the launch harness cannot bootstrap a fresh ledger (sampler needs the ledger the coordinator creates) | CL0822B table; classification report §10.3 | REVIEW | any fresh deployment or ledger reset | S172_BRIEF_I_PRODUCTION_SHAPE_CLASSIFICATION_REPORT.md:319-323 |
| Window-anchor **Brief II** (Optuna, registry, `distributed_config.json`, NPZ generation metadata, `WindowConfig.offset`, AC4; inherits mutant M8 100→149). Not written; blocked until Brief I is accepted. Then post-change acceptance, merge, D.1 differential experiment | CL0822A "Not done, deliberately"; TB_RULING_WINDOW_ANCHOR_SEQUENCING | REVIEW | window-anchor merge; D.1 | TB_RULING_BRIEF_I_PRODUCTION_SHAPE_FAILURE.md:6 ("Brief II remains BLOCKED"); `ls docs \| grep BRIEF_II` returns nothing |
| **Phase-7 soak**: 50-trial WATCHER run, `--start-step 1 --end-step 1`, `n_parallel=1`, 25 frozen identities; first at-scale run of the S166 in-memory clear (RAM series is the result) | FF §8, §2.54 | FLEET (owner launches) | S172 Phase 7 closure; manifest-default decision below | FF §2.54 (:2794-2840), §8 (:3111-3130). The 2026-08-04 soak failed (FF §2.54) |
| WATCHER failure-authority defect: after `Step 1 failed with code 1` WATCHER reported PASSED on file existence and triggered Step 2 | FF §2.34, §2.51 #9, §8 | REVIEW | Phase-7 autonomy ("Required before Phase-7 autonomy") | FF :1919-1925, :2588 |
| RANGE-MINER has no worker-launch mechanism; the fleet is hand-started with `scripts/launch_fleet_manual.sh` | FF §2.54, §8 | REVIEW | unattended operation | FF :2839-2840, :3114 |
| Decision owed after Phase-7 acceptance: manifest `default_params.use_range_miner` is `false` (PWC `use_persistent_workers: true` is still the manifest default); the manifest says not to enable until Phase 7 acceptance completes | found: agent_manifests/window_optimizer.json | REVIEW (after soak) | RANGE-MINER being the default WATCHER path | agent_manifests/window_optimizer.json:215, :261, :269 |
| rrig6600c runs Python 3.10.12 against 3.10.4 on the other two; a `pip freeze` diff across the three CTs settles it | FF §2.51 #1 | FLEET | "identical frozen clones" invariant | FF :2553-2558; STEP23 report :110-111 |
| F1/F2 heartbeat renewal path not certified by Beta | FF §2.51 #7 | REVIEW | — | FF :2578-2580 |
| Gate-12 attempt-5 initiating reader-exit cause unresolved | FF §2.51 #6 | REVIEW | — | FF :2576 |
| Red S172 suites: `test_s172_phase5_d6_production_adapter` 0/9 (stale fixture since F1); RC-1 — eight suites share the F1 claiming-model fixture mechanism; d0 / d1_workflow / d2 fixtures unmigrated. No RC-1 repair authorized | FF §2.51 #5; CL0822A finding 4 and AC7 section | REVIEW | a clean regression battery (AC7: 32 green / 15 red) | FF :2570-2574; CL0822A "Findings for Beta" 4, "AC7" |
| attempt-6 RXP-1/4 flake (one 77/78 in five runs) recorded, not patched | FF §2.51 #4 | REVIEW | — | FF :2566-2568 |
| `chapter1_p0` 8/12→10/12 improved with cause not established; flagged for its own investigation | CL0822A AC7 section | NONE | — | CL0822A ("an unexplained green is as suspicious as an unexplained red") |
| L-2 / coordinator ingress bounded by count, not bytes: add a coordinator RSS sampler on the next run; measure staging throughput vs payload; Beta ruling on a volume-aware bound; VM101 memory raise only after Brief I closes | LEADS L-2 = BL §20 | FLEET (sampler rides the next run); REVIEW for the ruling; OWNER for memory | knowing the safe survivor-volume region | miner/range_miner_coordinator.py:9602 per LEADS.md:96-100; LEADS.md:160-167; BACKLOG.md:520-534 |
| L-1 float32/float64 survivor-filter seam: sweep certified generations for affected boundaries; rule which side is intended; decide whether it shares a ruling with the §2.36 quantization item. Repair not authorized | LEADS L-1 = BL §21 | REVIEW | reproducibility of survivor populations at round thresholds | LEADS.md:77-88; BACKLOG.md:540-563 |
| WATCHER-MANIFEST-ROUTING-1 and the R2 recorded-but-unrepaired list: six-of-seven optimizer completeness check; `_ws_source` mislabels operator pins; `step_runner` is a separate uncertified dispatch surface | TB R2 ruling §8; CL0828 §7 | REVIEW | none blocked the commit | TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md §8 |
| Non-terminating multi-stripe loopback (`total_seeds = 2 × miner_stripe_size`): fixture limitation or production defect is not established; awaiting Beta ruling | BL §6 | REVIEW | — | BACKLOG.md:124-140; the only `TB_RULING*` hit for `loopback` is the S150 IPC request, which predates it |

### (b) Self-learning: selfplay, Chapter 13, LLM autonomy

| Item | Sources (first → last) | Waits on | Blocks | Anchor |
|---|---|---|---|---|
| Post-Step-6 autonomy trigger (4 list entries): call `dispatch_selfplay()` / `dispatch_learning_loop()` from `run_pipeline`, dispatch Chapter 13 from the WATCHER daemon, and test WATCHER→Ch13→selfplay. The dispatch functions, CLI and request scanning exist; `run_pipeline` has no such trigger. FF §0.6 defines Chapter 13 as "not a WATCHER step", so whether this wiring is still wanted needs a ruling | TM S120→S163KARG | REVIEW | — | agents/watcher_agent.py:2717-3114 (no selfplay/Ch13 call), :3189, :3834-3842; FF :167 |
| Part B task B3: `chapter_13_parameter_vocabulary.py` (auto-extract tunables from manifests). File absent; no manifest or policy read found in `chapter_13_llm_advisor.py`, although Phase-7 v3 says "already in advisor" | TODO_SELFPLAY B3; TODO_PHASE7 v3 :242 | REVIEW | Part B parameter application | `ls chapter_13_parameter_vocabulary.py` fails; grep of chapter_13_llm_advisor.py for `agent_manifests\|watcher_policies\|bounds` finds only prompt text (:23, :100) |
| Part B task B12 and its three preconditions: one threshold chokepoint; audit recommended / approved-applied / effective; make `watcher_policies.json` `parameter_application: true` truthful (advisory-only today) | TODO_SELFPLAY :105-131, B12 | REVIEW | LLM-proposed parameter changes taking effect | TODO_SELFPLAY_AND_LLM_AUTONOMY.md:112-131; watcher_policies.json:63; diagnostics_analysis_schema.py:73-74 |
| Bundle Factory Track 2: three retrieval functions return `[]` | TM S101→S148; TODO_PHASE7 v3 :267, :327; Ch13 progress v3_9 :141 | NONE | advisor context (outcomes, trends, incidents) | agents/contexts/bundle_factory.py:453-496 (three `return []`) |
| Feed `results/` trial history to the Strategy Advisor | TM S101→S127 | NONE | — | no reference in agents/contexts/bundle_factory.py or parameter_advisor.py (grep `trial_history\|results/`) |
| Phase 9B.3 proposal heuristics: `propose_transform_update` returns the transforms unchanged | TM S101→S163KARG; TODO_PHASE7 v3 :266, :326; FF §2.5 | REVIEW | selfplay proposing anything | policy_conditioned_episode.py:373-382; FF :378 |
| Selfplay promotion seam: ACCEPT does not change Step-1 parameters; `SelfplayCandidate` lacks `transforms` | TM S120→S127; FF §2.5 | REVIEW | selfplay affecting the pipeline | chapter_13_acceptance.py:221-232; PIPELINE_BEHAVIOUR_MODEL.md:677-680 |
| No opportunity trigger: every Chapter-13 trigger is degradation-based | FF §2.5; SM §6 | REVIEW | — | PIPELINE_BEHAVIOUR_MODEL.md:640-649; SM §6 |
| Selfplay NN enablement (2 entries): remove the neural_net guard in `inner_episode_trainer.py` and port y-normalization. The guard carries a stated reason ("500,000x worse MSE"), and SM §12.4 makes NN reinstatement conditional on gates NN-Q and NN-A | TM S145→S170 | REVIEW | NN in selfplay episodes | inner_episode_trainer.py:71, :497-501; SM §12 item 4 |
| Selfplay multi-model `train_best()` with NN (needs subprocess isolation) | TM S145 | REVIEW | — | TODO_MASTER_S145.md:82-86 |
| vmap batching for selfplay (deferred until NN is stable) | TM S150 | REVIEW | — | TODO_MASTER_S150.md:64 |
| Attribution activation P0-C: ranked-prediction artifact (no writer exists), canonical model loader in Ch13, NN scaler, A-NORM parity gate, signed contributions, one observe-only real-data cycle | found: SM §13 P0-C, §4.2 | NONE (design and loader work authorized; the real run needs data) | any valid attribution result; P1 | SM §4.2, §13, §14; chapter_13_orchestrator.py:582, :877; no `ranked_predictions.json` writer (grep) |
| P1: connect attribution to discovery (multi-draw evidence, advisory prior for selfplay) | found: SM §13 P1, §11 | REVIEW (authorized only after P0) | — | SM §13, §14 |
| Chapter 14 Phase 8 remainder: 8.4 marked in progress; tests 8.5–8.7 not run | TM S101→S147; Ch14 checklist | FLEET (Zeus + real data) | Phase 9 | CHAPTER_14_TRAINING_DIAGNOSTICS.md:3071-3074; chapter_13_orchestrator.py:652 |
| Chapter 14 Phase 9: first diagnostic investigation (9.1–9.6) | TM S101→S145; Ch13 progress v3_9 :131 | FLEET (Zeus + real data) | — | CHAPTER_14_TRAINING_DIAGNOSTICS.md:3076-3081 |
| Chapter 14 Phase 4: dashboard charts (4.1–4.8) | TM S101→S145 (as "web dashboard refactor"); Ch14 checklist | NONE | 9.3 | CHAPTER_14_TRAINING_DIAGNOSTICS.md:3034-3041; no `def chart_` in web_dashboard.py |
| Chapter 14 Phase 5: TensorBoard (optional) | TM S132; Ch14 checklist | NONE | — | CHAPTER_14_TRAINING_DIAGNOSTICS.md:3043-3048; no `SummaryWriter` in any .py |
| Chapter 14 task 3.6: catboost-with-diagnostics test | found: Ch14 checklist | FLEET (Zeus + real data) | — | CHAPTER_14_TRAINING_DIAGNOSTICS.md:3031 |
| `_record_training_incident()` is defined and never called | TM S101→S127 | NONE | retry audit trail for the advisor | agents/watcher_agent.py:1402 (only hit for the name in agents/) |
| Activate `draw_ingestion_daemon.py` on live draws | TM S120→S145 | OWNER | 24/7 feedback loop; waits on the scraper item | file present; TODO_MASTER_S145.md:46 |
| Scraper into a scheduler / 6-P2: `daily3_scraper.py --recent` overwrites the dataset; unit stays disabled until 6-P2 is certified; REV4 is a draft; no design document | TM S120→S145; CAT §7 gap 5; FF §8 | REVIEW | live draw ingestion | FF §2.9 (:531-547); CAT §7 gap 5; docs/TEAM_ALPHA_6P2_TRANSITION_RULING_REQUEST.md exists |
| 24-hour synthetic soak of the Chapter-13 loop (Soak C ran ~77 minutes) | TM S120→S145 | FLEET (Zeus) | going live | CHAPTER_13_IMPLEMENTATION_PROGRESS_v3_9.md:49-57 |
| WATCHER PRNG autonomy policy (switch `prng_type` between runs) | TM S114→S127 | REVIEW | — | only agent_grammars/sieve_analysis.gbnf:4 mentions `prng_type` |

### (c) Documentation and chapters

| Item | Sources (first → last) | Waits on | Blocks | Anchor |
|---|---|---|---|---|
| Audit the eleven never-audited chapters (4–14, ~11,000 lines) | BL §1; CAT §7 gap 6; FF §8 | NONE | trust in everything after the sieve | CAT §7 gap 6; BACKLOG.md:15-36 |
| Skill revision owed: v27.1 lacks EXEC-PIN-1, the proposed SR-3 (privileged seams), the stale-root-watcher fact, and still says Brief I does not exist and G-PROD-SHAPE never ran; third copy (ser8) unverified | BL §17, §23 skill note; CL0822A; CL0828 §4 | NONE (draft); OWNER installs the three copies | sessions loading a stale skill | FF :8-19 (currency v27.1); grep `EXEC-PIN-1\|SR-3\|_operator_pin_params` in the skill returns nothing; BACKLOG.md:623-630 |
| `TODO_TFM_FEATURE_CONTRACT_AND_ATTRIBUTION_ACTIVATION_v1_0.md`, `PROPOSAL_SELFPLAY_LEARNING_LOOP_REV2_1_ADDENDUM.md`, `PROPOSAL_TFM_ATTRIBUTION_DRIVEN_DISCOVERY_LOOP_v1_0.md` — named as the document set, none present | found: SM §15 | NONE | P0-A/B/C coding ("must exist before either broad proposal enters coding") | SM §15; `find . -name 'TODO_TFM*'` and `ls docs \| grep -iE 'REV2_1\|DISCOVERY'` return nothing |
| No committed document describes what `run_step3_full_scoring.sh` does (`STEP_SCRIPTS[3]` vs `full_scoring.json` divergence) | CAT §7 gap 2 | NONE | — | CAT §7 gap 2 |
| Step 0 (TRSE) has no chapter | CAT §7 gap 3 | NONE | — | CAT §7 gap 3 |
| Three binaries in `docs/` unread (two PDFs, `TEAM_BETA_REVIEW_kfolds_S100.docx`); the k-folds review may hold a ruling | CAT §7 gap 7 | NONE | the k_folds clamp item below | CAT §7 gap 7 |
| Chapter doc sync (Ch12, Ch13, Ch14, operating guide, README) by patch script | TM S120→S145; Ch13 progress v3_9 :137 | NONE | — | TODO_MASTER_S135.md:69; overlaps the chapter audit |
| `THRESHOLD_GOVERNANCE.md` still gives a 1K–10K survivor target | TM S145→S148 | NONE | — | docs/THRESHOLD_GOVERNANCE.md:69, :111; docs/CHAPTER_1_WINDOW_OPTIMIZER.md:594 |
| Chapter 2 VIR-6 residual: bodies of the 40 non-java_lcg kernels unread | Ch2 §14.3 | NONE | — | CHAPTER_2 §14.3 last row |
| Chapter 2 F-7: whitepaper `G(s,−i)` vs forward-against-reversed-residues, named not resolved | Ch2 §12, §14.3 | REVIEW | — | CHAPTER_2 §12 F-7 |
| No `TB_RULING_*` file for the 2026-07-30/31 session-stream rulings | Ch2 §12.1 | OWNER (ruling text is outside the repo) | — | `ls docs \| grep -iE 'TB_RULING.*(SESSION\|DATASET\|LIFECYCLE\|STREAM)'` returns only an S172 Phase-4 request |
| Gate-12 brief corpus lives in `~/dashboard_work`, not the repository | CAT §7 gap 9 | OWNER | traceability from a clone | CAT §7 gap 9 |
| ser8 pre-repository archive never inventoried | CAT §7 gap 13; FF §8 | OWNER (credential) | ~20-changelog backfill and the S-numbering decision (SR-2) | CAT §7 gap 13 |
| `agent_manifests/definitions.json` untracked, gitignored, role undocumented | CAT §7 gap 4 | OWNER (file is on the host) | — | `git ls-files agent_manifests` lists 7 files, none is definitions.json |
| Four arms of the Chapter-1 executable gate need a reachable fleet | Ch1 §17.3 | FLEET | — | CHAPTER_1 §17.3 last row |

### (d) Downstream pipeline, Steps 2 to 6

| Item | Sources (first → last) | Waits on | Blocks | Anchor |
|---|---|---|---|---|
| Run Step 2 on current survivors. Through WATCHER, `STEP_SCRIPTS[2]` reaches `run_scorer_meta_optimizer.sh`, which calls the prohibited legacy converter and `mv`s a regular file onto the finalizer-owned symlink | TM S163→S163KARG; FF §8 | REVIEW | Steps 2–6 on RANGE-MINER output via WATCHER | run_scorer_meta_optimizer.sh:87, :97; FF :3121-3125 |
| D3.0-B: `convert_survivors_to_binary.py` fabricates `java_lcg` for records with no identity; Beta ruled OPEN, requires completion; not scoped | BL §12; FF §2.18, §8 | REVIEW | any use of the legacy converter | convert_survivors_to_binary.py:184; BACKLOG.md:250-292 |
| `forward_matches` / `reverse_matches` are in the NPZ and absent from the Step-3 merge list; needs a governed schema decision (Option A or B). Same item as "S103 Part 2" | TM S120→S170 (S103 Part 2); BL §1; Ch2 §12.1; SM §7.3 | REVIEW | per-seed sieve signal reaching the model | survivor_scorer.py:774-779; SM §7.3 item 3 |
| P0-A: canonical 91-feature extraction and 89-feature model-input manifests; remove duplicated static feature lists | found: SM §13 P0-A, §9 | NONE | P0-C; honest importance and drift tooling | SM §9, §13, §14 ("Authorized") |
| P0-B: Step-3 semantic parity — exact metadata guard, identical batch and sequential merges, F-PARITY tests, fail closed | found: SM §13 P0-B, §7.3 | NONE | training on fallback output (Beta: P0) | generate_step3_scoring_jobs.py:95-100; SM §7.3 items 1–2; no `F-PARITY` in any .py |
| Step-3 output validation floor is `< 46` against a 91/89 contract | BL §15 | NONE | — | run_step3_full_scoring.sh:475-478 |
| `full_scoring.json` declares 26 GPUs; frozen set is 25 (documentation drift) | BL §16 | NONE | — | agent_manifests/full_scoring.json:105-109 |
| `java_lcg_cpu` non-zero-skip reachability audit (read-only; no fix authorized) | BL §8.1; Ch2 §12.1; FF §8 | NONE | a correct known-answer reference | survivor_scorer.py:124; full_scoring_worker.py:305 |
| Skip-output work: first settle whether a hit mask reaches the host; then restore `skip_mean/std/entropy` behind a validity mask under a schema-version decision for the five dead placeholders | BL §2; SM §8.3 | NONE (the hit-mask question); REVIEW for the schema change | three dead features | window_optimizer_integration_final.py:140-151; BACKLOG.md:60-63; SM §8.3 |
| DEP-ABI-V2-NPZ-SEMANTICS: audit every consumer of canonical `offset` before nonzero generator phase is allowed | BL DEP-ABI-V2 | NONE (read-only audit) | ABI v2 / nonzero phase | BACKLOG.md:453-491 |
| k_folds deeper clamp `max(3, n_train // 3000)` (the S124 guard is in; this is not) | TM S101→S163 | REVIEW (list requires TB review) | — | meta_prediction_optimizer_anti_overfit.py:2026; no `n_train // 3000` in the file |
| Phase 3B: parallel tree trials (parallel Optuna exists for neural_net only) | TM S101→S148 | REVIEW (list requires TB review) | Step-5 wall clock | meta_prediction_optimizer_anti_overfit.py:1746-1751 |
| Phase 3A vmap: k_folds scaling test and benchmark on real data (2 entries) | TM S101→S145 | FLEET (Zeus + real survivors) | — | TODO_MASTER_S145.md:130 |
| sklearn feature-name warnings in Step 5 | TM S120→S170 | NONE | — | no `filterwarnings` in meta_prediction_optimizer_anti_overfit.py |
| Low-variance warning in the legacy converter ignores window size | TM S163 | REVIEW (legacy converter is under D3.0-B) | — | convert_survivors_to_binary.py:189-193 |

### (e) Infrastructure and hygiene

| Item | Sources (first → last) | Waits on | Blocks | Anchor |
|---|---|---|---|---|
| `.s172_accumulator/generations/` has no backup policy (unowned) | BL §19; FF §2.51 #8, §8 | OWNER | only copy of certified generations is VM101 | BACKLOG.md:421-449 |
| D6.3: `.s172_checkpoint/<run_id>/` never pruned (~10.7 KB/run, non-blocking) | FF §2.9, §8 | REVIEW | — | FF :528-531 |
| `dataset_provenance/*.json` never pruned | BL §7; FF §2.9 | REVIEW | — | BACKLOG.md:144-149 |
| Three `[WATCHER][RETRY]` lines still announce changes the step filter may drop | BL §4 | REVIEW (file is the certified Run-4 digest) | misreading a soak log | agents/watcher_agent.py:2378, :2382, :2386 (BACKLOG cites the old :1725-1733) |
| `.gitignore:42` dead negation; stale NPZ negation removable as hygiene | BL §9, §19 | REVIEW | — | .gitignore:42; BACKLOG.md:439-440 |
| `_RusageChildrenSampler` reads a process-lifetime high-water mark; scope-correct fix flagged, not actioned | BL §11; FF §2.9 | REVIEW (edits a certified D4/D5 arm) | any new D5 arm reaping a large child | tests/test_s172_phase5_d5_process_sharded.py:2107-2119 |
| Stale root-level `watcher_agent.py` (72,042 B) beside the live `agents/watcher_agent.py`; deletion not proposed, Beta's ruling | BL §23; CL0828 §7 | REVIEW | — | root file size confirmed 72042; BACKLOG.md:596-621 |
| Root directory clutter ("S110 root cleanup"); standing ruling leaves stale duplicates alone absent a ruling | TM S120→S170 | REVIEW | — | 836 root entries (`ls \| wc -l`); BACKLOG.md:619-621 |
| `results/` FIFO pruning and post-run hook (2 entries) | TM S101→S145 | REVIEW (retention policy) | — | TODO_MASTER_S145.md:150 |
| Post-run JSON cleanup | TM S170 | REVIEW (retention policy) | — | TODO_MASTER_S170.md:92-93 |
| Fallback parity pass 2 (needs `.127` booted; Zeus runs one OS at a time); pytest 9.1.1 not yet in a committed env artifact | CAT §7 gap 12; CL0828 §9 | OWNER | fallback box parity | CAT §7 gap 12; CL0828 "Fallback parity" |
| SR-1 analog for the pinned-executable-source vs live-helper coupling hazard: recommended, not enacted | CL0822A finding 3 | REVIEW | — | CL0822A "Findings for Beta" 3 |
| ROCm prelude hostname guard is dead on every live rig (`rig-6600*` vs `rrig6600*`); do not rename hosts; any repair keys from a profile property | Ch2 §12 F-5 | REVIEW | — | sieve_filter.py:26 |
| Node watchdog / auto-restart layer. Constraint since: a rebooted rig returns as Ubuntu, not Proxmox, and no agent reboots a host unasked | TM S120→S127 | OWNER | unattended 24/7 | FF :8 (v27.1 boot-selector rule) |
| Web dashboard auto-start with WATCHER | TM S163KARG | OWNER (dashboard work is host state) | — | TODO_MASTER_S163KARG.md:49-51 |
| `run_status.json` push or status endpoint for remote polling | TM S163KARG | OWNER | — | TODO_MASTER_S163KARG.md:78-80 |
| Telegram alert on worker quarantine / rig reboot (boot notify exists; quarantine alert not found) | TM S152→S154 | REVIEW | — | SESSION_CHANGELOG_20260719_TELEGRAM_PROXMOX.md:8-20 covers boot only; execution_set.py:17 says boot notify is Telegram-only |

### (f) Other

| Item | Sources (first → last) | Waits on | Blocks | Anchor |
|---|---|---|---|---|
| Hybrid kernels ignore sampled `skip_min`/`skip_max` (`expected_skip = 5`); remedy is wire-in, not removal; decision open | FF §2.7 #4; Ch1 §17.3 D-1/D-2; BL §3 | REVIEW | hybrid certification; sampler comparison | FF :403; CHAPTER_1 §17.3 row 1 |
| Certifying TPE-vs-random sampler comparison waits on hybrid skip-bound semantics or a phase-aware search space | BL §3; FF §8 | REVIEW | autonomous sampler selection | BACKLOG.md:71-92 |
| `recommended_window_size` is read and never used; Rule A uses literal 32 | FF §2.7 #5b | REVIEW | — | window_optimizer_bayesian.py:609; FF :405 |
| `skip_learning_rate` configured 0.2–0.7, kernel adapts at 1.0 | FF §2.7 #6 | REVIEW | — | FF :406 |
| Sampler provenance guard (fail before study) required before another sampler is registered | BL §8.2; Ch1 §17.3 | REVIEW (no fix authorized) | BoTorch or any new sampler | window_optimizer_bayesian.py:925-927; BACKLOG.md:179-191 |
| BoTorch dual-GPU sampler | TM S170 | REVIEW | — | TODO_MASTER_S170.md:64-72; blocked by the provenance guard |
| NP2 checkpoint transaction design for `n_parallel > 1` | BL §13; FF §8 | REVIEW | Optuna parallelism above 1 | BACKLOG.md:296-301 |
| Session-separated dataset authority artifact; includes the list's "per-segment pipeline runs" (per-session is now the rule, combined is prohibited by default) | BL §5; TM S120→S148 | REVIEW | certifiable per-session runs | BACKLOG.md:104-120 |
| Combined-session mode is still offered first and sampled | Ch1 §17.3 | REVIEW | autonomous runs landing on an uncertifiable configuration | window_optimizer.py:182-186 |
| `run_with_config` writes `[]` and prints success | Ch1 §17.3 | NONE | — | CHAPTER_1 §17.3 row 6 |
| TRSE `except` handler calls `logger`, not defined at module level | Ch1 §17.3 | NONE | — | window_optimizer.py:1084 — the only occurrence of `logger` in the file |
| `min_workers` defaults: 1 in the signature, 24 at the CLI | Ch1 §17.3 | REVIEW | — | window_optimizer.py:1147, :1501 |
| `args_map` maps two fail-closed threshold flags (dormant) | Ch1 §17.3 | OWNER (manifest owner) | — | CHAPTER_1 §17.3 row 9 |
| `parameter_bounds` admits `window_size` 2 where the S172 search floor is 6 | Ch1 §17.3 | REVIEW | — | agent_manifests/window_optimizer.json parameter_bounds window_size (2, 50); agents/step_runner/command_builder.py:151-170 |
| `--rig-profile` / `--execution-set-nodes` not in `args_map` | Ch1 §17.3 | OWNER (manifest owner) | WATCHER selecting a partial fleet | CHAPTER_1 §17.3 row 11 |
| `GridSampler` unconstructible at live bounds; design decision | Ch1 §17.3; FF §2.9 | REVIEW | — | CHAPTER_1 §17.3 row 12 |
| Search-strategy repair (`random`/`grid`/`evolutionary` gated at CLI) pending Beta ruling Q3 | Ch1 §17.3; FF §2.9 | REVIEW | — | CHAPTER_1 §17.3 row 13 |
| TRSE Rules B and C activation decision (logged only) | TM S122→S148 | REVIEW | — | window_optimizer_bayesian.py:632-641 |
| TRSE `confirmed_windows` accumulation analysis | TM S122 | OWNER (needs run artifacts) | Rule A refinement | TODO_MASTER_S122.md:55 |
| `2019-01-25` evening-only record | BL §14 | OWNER (needs the dataset and the source site) | — | BACKLOG.md:305-310 |
| Battery Tier 1B and related features (7 entries): linear complexity, lag 12/25/27 autocorrelation, bit avalanche, bit-plane runs, spectral lattice, real-vs-synthetic battery comparison, binary matrix rank. Any new feature is a schema-version change | TM S114→S148 | REVIEW | — | none of the names in survivor_scorer.py (grep); SM §8.3, §13 |
| PRNG registry experiments (2 entries): run Step 1 with xorshift variants; `java_lcg_xor` | TM S114→S127 | OWNER | — | no `java_lcg_xor` in prng_registry.py |
| Digit-triplet frequency drift probe | TM S120→S127 | OWNER (needs data) | — | TODO_MASTER_S127.md:114 |
| PA Pick 3 follow-up sweep | TM S143→S148 | OWNER | — | pa_pick3_scraper.py tracked; dataset not in the clone |
| Threshold calibration follow-ups (3 entries): `phase2_threshold`, other PRNG families, real draws | TM S148 | FLEET | — | TODO_MASTER_S148.md:53-58 |
| Post-sweep analyses (4 entries): yield decay, seed distribution, quality vs seed range, sufficiency threshold | TM S145→S148 | OWNER (needs run data) | — | TODO_MASTER_S148.md:116-119 |
| Walk-forward simulation | TM S148→S154 | OWNER | — | TODO_MASTER_S154.md:116 |
| Add throughput metrics to Optuna trial metadata | TM S125b | REVIEW | — | TODO_MASTER_S125b.md:22 |
| Threshold optimization heatmap (Display 2.1) and other post-run plots (2.2) (2 entries) | TODO_DISPLAY 2.1, 2.2 | NONE | — | no `visualization/` directory; web_dashboard.py:49 is a feature-importance heatmap, not thresholds |
| Results-system integration remainder (Nov 2025 roadmap): `--legacy-format` flag, direct `window_optimizer.py`, sweep, pattern tools, `unified_system` reader and menu | INTEGRATION_ROADMAP.txt | REVIEW (whether still wanted) | — | INTEGRATION_STATUS_REPORT.txt:40-67; no `sweep_adapter.py` in integration/ |

### Duplicates across sources

Each of these is one row above. The sources describe the same thing in different words.

| Item | Appears in |
|---|---|
| `forward_matches` / `reverse_matches` not reaching the model | TM "S103 Part 2" (S120→S170) · BL §1 · Chapter 2 §12.1 and §14.3 · SM §7.3 and P0-B item 5 |
| Unaudited chapters | BL §1 (names 5) · CAT §7 gap 6 (names 11) · FF §8 · TM "chapter docs patch script" overlaps it |
| `java_lcg_cpu` non-zero-skip mismatch | BL §8.1 · Chapter 2 §12.1 · FF §8 Wall C caution |
| Sampler provenance guard | BL §8.2 · Chapter 1 §17.3 · FF §2.9 |
| Hybrid `skip_min`/`skip_max` dead at the kernel | FF §2.7 #4 · Chapter 1 §17.3 D-1/D-2 · BL §3 (as the blocker of the sampler comparison) |
| `offset` fusion (F-4 / C-2 / D-3 / FF §2.7 #5) | Chapter 1 §17.3 (two rows) · Chapter 2 §12 and §14.3 · FF §2.7 #5 · now the Brief I acceptance row |
| Coordinator ingress byte bound | LEADS L-2 · BL §20 · TB R2 ruling §8 |
| float32/float64 seam | LEADS L-1 · BL §21 · TB R2 ruling §8 |
| Accumulator backup policy | BL §19 · FF §2.51 #8 · FF §8 |
| D3.0-B | BL §12 · FF §2.18 · FF §8 |
| NP2 checkpoint design | BL §13 · FF §8 · TM "n_parallel partition design" is its superseded ancestor |
| D6.3 and `dataset_provenance` retention | FF §2.9 · FF §8 · BL §7 |
| Scraper / 6-P2 | TM "wire scraper into WATCHER scheduler" · CAT §7 gap 5 · FF §2.9 and §8 |
| Fallback parity pass 2 | CAT §7 gap 12 · footer of CL0822A, CL0822B, CL0828 |
| ser8 archive inventory | CAT §7 gap 13 · FF §8 · behind TM "write S159 changelog" (UNKNOWN) |
| Stale root `watcher_agent.py` | BL §23 · TB R2 ruling §8 · TM S147 notes the stale root copy · TM "S110 root cleanup" is the general case |
| Skill currency | CAT §7 gap 10 (closed at v27) · BL §17 · BL §23 skill note · CL0822A (EXEC-PIN-1 not recorded) |
| Bundle Factory Track 2 stubs | TM S101→S148 · TODO_PHASE7 v3 deferred table · Chapter 13 progress v3_9 |
| Phase 9B.3 heuristics | TM S101→S163KARG · TODO_PHASE7 v3 · FF §2.5 (`propose_transform_update` no-op) |
| Chapter 14 phases | TM (with its own, different phase numbers until S132) · the chapter's checklist · Chapter 13 progress v3_9 |
| Dispatch of selfplay / learning loop | TODO_SELFPLAY B13/B14 and TODO_PHASE7 v3 Part B (done) · TM S120→S163KARG (a different, post-Step-6 trigger, open) |
| Selfplay Part A tests | TODO_SELFPLAY A1–A7 · TODO_PHASE7 v3 Part A · TM S120 completed table |
| Dead skip features | BL §2 · SM §8.3 · ZMQ `skip_sequences` entry in TM S159→S163 |
| Per-session runs | TM "per-segment pipeline runs" · BL §5 · Chapter 1 §17.3 combined-session row |
| WATCHER failure-authority defect | FF §2.34 · FF §2.51 #9 · FF §8 · CL 2026-08-20 §5 |
| First observation of `_bp` falsifiers | FF §2.51 #3 and §2.54 (still "owed") · TB ruling of 2026-08-22 (recorded) |
| Netconsole re-arm | LEADS L-3 follow-up · CL0828 §7 (two rows) · TB R2 ruling step 3 |

## 4. UNKNOWN items

Could not be determined from a public clone.

| Item | Sources | What was searched, and what would settle it |
|---|---|---|
| `docs/SESSION_CHANGELOG_20260819_S1.md` — untracked, unattributed, S-numbered against SR-2; owner disposition pending | CAT §7 gap 1; CL 2026-08-20 §5 | Not in the clone (`ls docs` shows 20260728_S1 and 20260801_S1 only). Settles on the host: `git status --porcelain docs/` on VM101 and the owner's decision |
| Whether the S179-authorized KPI governance implementation landed | CAT §7 gap 8 | `grep -ci kpi agents/watcher_agent.py` = 0; six `watcher_kpi_*.py` analyzers exist at root. The implementation may use other names. Settles by reading TB_RULING_S179_IMPLEMENTATION_AUTH.md's three code conditions and tracing each in WATCHER / Chapter 13 |
| Two doc-generator defects "carried forward, not re-characterised" | BL §9 | BACKLOG.md:208 does not say what they are. Settles by finding the original statement (search the 2026-08-01…03 changelogs and TEAM_ALPHA submissions for "doc-generator") |
| Chapter 2 F-3: is the three-lane redundancy residue of an unbuilt lane-parallel CRT design | Ch2 §12, §14.3 | Chapter says not determinable from available surfaces. Settles from the ser8 pre-repository archive (CAT §7 gap 13) |
| Regression diagnostic gate → True | TM S120→S163KARG | `gate=True/False` appears only in test_phase_8_soak.py:558-563; the production switch was not found by that name in tracked .py files; config JSON is gitignored. Settles by tracing `_detect_hit_regression` consumers in chapter_13_orchestrator.py and the host's watcher_policies / config |
| TODO_SELFPLAY A7: `chapter_13_acceptance.py --validate-selfplay` test | TODO_SELFPLAY A7 | CLI exists (chapter_13_acceptance.py:945); no record of the run found in TODO_PHASE7 v3 (A1–A5 only) or docs grep. Settles from the S57–S59 / S116 changelogs or by running it |
| Multi-trial dispatch (Step 5) | TM S101→S114 | One line, never defined, dropped at S120 with no completion note. Settles from the S100/S101 changelogs (not found in the clone; may be on ser8) |
| Write SESSION_CHANGELOG for S159 | TM S159 | Only `SESSION_CHANGELOG_20260402_S159G.md` is present. ~20 changelogs are ser8-only (FF §1.3 SR-2). Settles from the ser8 backfill |
| Claude Project housekeeping (2 entries): remove 27 stale files; upload updated docs / stale files | TM S101→S163; Ch13 progress v3_9 :136 | The Claude Project is outside the repository. Settles by the owner checking the project |
| Display Phase 1 live items 1.2–1.5 (GPU utilization, trial progress, survivor counts, ETA) | TODO_DISPLAY 1.2–1.5 | progress_display.py:74-112 has trial/best/ETA fields; web_dashboard.py exists but has 0 references to the miner. RANGE-MINER-era dashboard work is in `~/dashboard_work` (host). Settles by inspecting the running dashboard on VM101 |
| Rig host settings from S154 (3 entries): disable syncthing, 8 GB swap, `vm.overcommit_memory` | TM S154 | Host state on rigs that have since moved to Proxmox CT100 clones (SESSION_CHANGELOG_20260719_TELEGRAM_PROXMOX.md:3-5). Settles on the rigs when powered |
| DPM harness (900 mV / 2100–2200 MHz) as a persistent service | TM S170 | TM S163KARG records an `amdgpu-dpm-pin.service` on all rigs; R2 step 3 now asks for "actual GPU power caps" to be verified. Whether the harness was superseded by the pin is host state. Settles with the fleet up |
| `crash_forensic_daemon.py` three bugs | TM S163KARG→S170 | File is not in the clone (`find . -name 'crash_forensic*'` empty). Settles on VM101 / Zeus |
| rig crash monitors deployed before launch | TM S152→S154 | `rig_crash_monitor.sh` is not in the clone; netconsole is the current fault surface (LEADS L-3). Settles on the rigs |

## 5. SUPERSEDED items

"PWC-only" was checked, not assumed: the `miner/` package's imports (read in all nine modules) include none of `persistent_worker_coordinator`, `sieve_gpu_worker`, `pwc_*` or `zmq_*`, and FF §0.7 / :559 record PWC and ZMQ as retired from certifying use.

| Item | Sources | Superseded by | Anchor |
|---|---|---|---|
| Verify S101 full run results | TM S101→S114 | later real-data runs | agents/watcher_agent.py:169 ("real data yields 85"); TODO_MASTER_S120.md:4 |
| Step-5 `n_jobs` investigation ("hardcoded to 2") | TM S101→S114 | S95 design: `n_jobs` = detected GPUs for neural_net, 1 otherwise; remaining question is Phase 3B | meta_prediction_optimizer_anti_overfit.py:1746-1751 |
| `--save-all-models` flag | TM S101→S135 | withdrawn: S170 lists it under "Do NOT Include in TODO Lists" | TODO_MASTER_S170.md:134-136 |
| Chapter 14 "Phase 5: FIFO history pruning" | TM S101→S127 | numbering: the chapter's Phase 5 is TensorBoard; S132 re-audited the list | CHAPTER_14_TRAINING_DIAGNOSTICS.md:3042; TODO_MASTER_S132.md:45 |
| MultiModelTrainer inline path "dead code" comment-out | TM S101→S145 | premise no longer holds: the inline path is the documented fallback and the tree path | meta_prediction_optimizer_anti_overfit.py:370, :1178-1180, :2112 |
| Always launch pipeline in tmux | TM S101→S143 | invariant "always use nohup, never tmux" | TODO_MASTER_S145.md:203 |
| NN needs 500+ survivors | TM S120→S127 | survivor volume is no longer the limit (13.1M phase-3 survivors in run 3) | BACKLOG.md:508-510 |
| Fix Zeus double-dispatch (`n_parallel=2` partition design) | TM S120→S127 | built S125, then `n_parallel=2` scrapped; `n_parallel=1` is binding and NP2 is separate new work | TODO_MASTER_S163KARG.md:24; BACKLOG.md:296-301; manifest `n_parallel: 1` |
| Run 200-trial Step 1 / monitor 200-trial study / update `search_bounds` from it (3 entries) | TM S120→S143 | production sweep, then RANGE-MINER; search floor since set by S172 ruling | TODO_MASTER_S145.md:11-20; CHAPTER_1 §17.3 row 10 |
| Force-refresh `trse_context.json`; first full 0→6 run with TRSE (2 entries) | TM S122, S135 | TRSE context is read by every Step-1 command since | CL0828 §5 (`--trse-context trse_context.json` in the built command) |
| Verify NPZ accumulator fix end-to-end | TM S139→S143 | D3.5 finalizer and per-generation accumulator | BACKLOG.md:426-432 |
| Production sweep on PWC (9 entries): recalc timing, smoke-test script fix, launch Run 1, resume Run 1, commit NPZ to git, sweep_run2, sweep_run3, sweep_run4, window-12 decision | TM S145→S159 | RANGE-MINER Gate-12 runs; NPZ payloads must not go back into git; Run 4 uses k=12 | FF §0.7 (:181-189); BACKLOG.md:436-438; s145r1_smoke_tests.sh:32 still tests the JSON |
| Q0 zero-survivor branch proof; hybrid forward prune-rate monitoring; Q2 refinement phase (3 entries) | TM S147→S148 | these live in the PWC `run_trial_persistent` path; miner package imports none of it | miner/*.py import lists (no `persistent_worker_coordinator`, `sieve_gpu_worker`, `pwc_*`, `zmq_*`) |
| Fix `soak_s130.sh` | TM S147→S148 | PWC soak; PWC is non-certifying | soak_s130.sh:46-47; FF :188-189 |
| Z10 kernel missing from `sieve_gpu_worker.py` | TM S135 | PWC worker only | miner import lists (above) |
| rrig6600c throughput deficit weighting | TM S135 | pull model: daemons pull stripes | FF §2.50 (:2529-2533) |
| Optuna node-failure resilience | TM S132→S148 | F1/F2 active-lease scheduler on the sieve path. Note: no storage-error retry exists in window_optimizer_bayesian.py (grep) | FF §2.27 (:1523-1540) |
| Persistent-worker session drops on AMD rigs; investigate drops after extended runs (2 entries) | TM S139→S152 | PWC SSH/TCP sessions | FF §0.7; §2.50 |
| Write survivors to disk as found | TM S152 | coordinator staging per sub-stripe plus D6.2 checkpoint | FF §2.19 (:1019-1029); FF §2.9 (:503-512) |
| rrig6600c sshd verification, MaxSessions investigation, slim_v1 re-enable, sshd check in PWC preflight, sshd in the node checklist (5 entries) | TM S152 | PWC over SSH | FF §0.7 |
| Circuit breaker v3; respawn-storm fix; OOM preflight check; syncthing preflight check (4 entries) | TM S152→S154 | PWC respawn / preflight | FF §0.7; §2.50 |
| Remove legacy dict-list coordinator parser; GPU-utilization measurement of Pass 3/4 (2 entries) | TM S150→S154 | PWC; and standing ruling leaves stale code alone absent a ruling | BACKLOG.md:619-621 |
| `java_lcg_reverse` kernel arg mismatch patch | TM S154 | miner worker builds the 12-argument reverse call itself; the S153 patch file is not in the clone | miner/range_miner_worker.py:333-334 |
| ZMQ validation ladder (6 entries) | TM S159 | completed S159–S160, then ZMQ retired from certifying use | TODO_MASTER_S163.md:11; FF :559 |
| Raise `seed_cap_amd` 2M→10M | TM S159 | stripe sizing (`miner_stripe_size` 2^26); raising launch count per trial is itself constrained | FF §2.50 (:2538-2541) |
| ZMQ cleanup hardening; ZMQ `skip_sequences` empty-list check (2 entries) | TM S159→S163 | ZMQ retired; the `skip_sequences` question continues as BL §2 | FF :559; BACKLOG.md:40-49 |
| `free_all_blocks()` removal in `sieve_gpu_worker.py` | TM S163→S170 | PWC worker only (still present there) | sieve_gpu_worker.py:91-92 |
| Zeus `cudaErrorDevicesUnavailable` proposal | TM S163 | Zeus local PWC sieve path | miner import lists |
| Pre-warm CuPy kernel cache; TCP worker persistence across trials; ephemeral coordinator benchmark (3 entries) | TM S163 | persistent per-GPU daemons | FF §0.7 (:184-186) |
| Zeus TCP worker path; TCP-PWC job pre-fetch; ZMQ job pre-fetch (3 entries) | TM S163KARG→S170 | pull-model daemons | FF §2.50 |
| Stability curve test for `seed_cap_amd` | TM S170 | PWC stabilisation; FF names S162–S170 pacing work as band-aids replaced by RANGE-MINER | FF §2.50 (:2543-2547) |
| rrig6600 "script write failed" investigation | TM S170 | PWC-TCP transport | FF §0.7 |
| rrig6600c `GCVM_L2_PROTECTION_FAULT` root cause | TM S170 | attributed to the PWC launch storm; avoided, not eliminated. The soak's GCVM criterion reports UNAVAILABLE, never PASS | FF §2.50 (:2533-2541); FF §2.17 (:890-905) |
| TODO_SELFPLAY B15: wire the LLM advisor into the WATCHER daemon | TODO_SELFPLAY B15 | advisor is called by the Chapter-13 orchestrator; WATCHER consumes its requests | chapter_13_orchestrator.py:51, :386; agents/watcher_agent.py:3189 |
| TODO_SELFPLAY B16–B19: tests via `watcher_agent.py --validate-proposal` | TODO_SELFPLAY B16–B19 | no such CLI; the accept / reject / escalate cycle was exercised by Soak C (81 cycles, 73 auto-executed, 6 rejected) | CHAPTER_13_IMPLEMENTATION_PROGRESS_v3_9.md:49-57 |
| TODO_PHASE7 (REVISED) open Part B / Part D boxes | TODO_PHASE7 REVISED | v3 of the same list, all parts complete | TODO_PHASE7_WATCHER_INTEGRATION_REVISED_v3.md:5 |

## 6. DONE items

| Item | Sources | Evidence |
|---|---|---|
| k_folds `n_splits <= n_samples` guard | TM S132 | meta_prediction_optimizer_anti_overfit.py:2026 |
| Ch14 Phase 6 WATCHER integration; Phase 7 LLM integration | TM S101→S127 | CHAPTER_14_TRAINING_DIAGNOSTICS.md:3050-3066; diagnostics_llm_analyzer.py, grammars/diagnostics_analysis.gbnf present |
| Feature names in `best_model.meta.json` | TM S101→S148 | meta_prediction_optimizer_anti_overfit.py:587-593, :2903 (under `feature_schema`; Ch13 reads the wrong key — SM §4.2 blocker 2) |
| WATCHER survivor threshold 100→50 | TM S120→S145 | agents/watcher_agent.py:169-172 |
| Manifest fix (`enable_pruning`, `n_parallel`, study flow) committed | TM S120→S127 | agent_manifests/window_optimizer.json:5 (v1.8.0), :33-34, default_params |
| NN y-label normalization (Step 5) | TM S120→S143 | train_single_trial.py:504-510, :689-690 |
| XGBoost device-mismatch fix | TM S120→S148 | models/wrappers/xgboost_wrapper.py:231-245 |
| Remove CSV writer from `coordinator.py` | TM S120→S170 | no `csv` in coordinator.py (grep); still listed in S170 |
| Variable-skip bidirectional count in the Optuna score | TM S125b→S148 | window_optimizer_integration_final.py:1888, :1945-1950; still listed in S148 |
| GPU throughput probes and seed-cap update (8 entries) | TM S125b→S127 | scripts/probe_phase_A_amd.sh, probe_phase_A_rtx.sh, probe_phase_C_stability.sh, scripts/apply_caps.py; TODO_MASTER_S132.md:89-95 |
| WATCHER Step-1 timeout | TM S139→S145 | agents/watcher_agent.py:3575 |
| `--force-step N` | TM S139→S152 | agents/watcher_agent.py:1768-1771, :3537-3543 |
| Remove hardcoded W8_O43 warm start | TM S143 | window_optimizer_bayesian.py:769 |
| Telegram boot-notify hardening on all rigs | TM S120→S127 | SESSION_CHANGELOG_20260719_TELEGRAM_PROXMOX.md:8-20 |
| CuPy memory-pool OOM fix | TM S154 | sieve_gpu_worker.py:397-425; persistent_worker_coordinator.py:120-123 |
| SESSION_CHANGELOG for S152 | TM S154 | docs/SESSION_CHANGELOG_20260322_S152.md |
| Manifest: `pwc_transport` default tcp; `use_zmq_sqlite` default and docs in sync; transport flag | TM S159→S163KARG | agent_manifests/window_optimizer.json:42-43; default_params `pwc_transport: tcp`, `use_zmq_sqlite: false`; param_docs default false |
| TODO_PHASE7 v3 Parts A, B0, B, C, D | TODO_PHASE7 v3 | v3 :57-128; agents/watcher_dispatch.py; agents/watcher_agent.py:3834-3846 |
| Soak tests A, B, C | TODO_PHASE7 v3 :293-301 | CHAPTER_13_IMPLEMENTATION_PROGRESS_v3_9.md:49-57 |
| Parameter advisor (Item B) | TODO_PHASE7 v3 :265, :325 | parameter_advisor.py; chapter_13_orchestrator.py:840; Ch13 progress v3_9 :112 |
| TODO_SELFPLAY A1–A6 | TODO_SELFPLAY Part A | TODO_PHASE7_WATCHER_INTEGRATION_REVISED_v3.md:57-65 |
| TODO_SELFPLAY B1, B2, B4–B11, B13, B14 (files and functions exist) | TODO_SELFPLAY Part B | llm_proposal_schema.py; agent_grammars/chapter_13.gbnf; chapter_13_diagnostics.py; chapter_13_llm_advisor.py; chapter_13_acceptance.py:327; agents/watcher_agent.py:3834-3842 |
| TODO_SELFPLAY B20 selfplay dispatch dry run | TODO_SELFPLAY B20 | TODO_PHASE7 v3 :125 |
| D6 threshold propagation precondition for Part B | TODO_SELFPLAY :82-110 | tests/test_s172_phase5_d6_threshold_path.py; miner/range_miner_coordinator.py:9061, :9117; FF :400 |
| Display 1.1 auto-attach progress monitor (legacy coordinator path) | TODO_DISPLAY 1.1 | coordinator.py:129-149 |
| Results system: coordinator integration | INTEGRATION_ROADMAP week 2 | coordinator.py:1874-1881; integration/coordinator_adapter.py |
| CA draw-procedures PDF in the repo | BL §9; Ch2 §12.1 | docs/reference/CA_DAILY_SLP_DRAW_PROCEDURES_20210609.pdf |
| Skill stale at v26 | CAT §7 gap 10 | FF :8 (v27.1); SESSION_CHANGELOG_20260820_SKILL_V27.md. A new revision is owed — see group (c) |
| Window-anchor Brief I does not exist | CAT §7 gap 11; FF §8 | docs/S172_WINDOW_ANCHOR_BRIEF_I.md; CL0822A; legacy routes closed at coordinator.py:835, distributed_worker.py:292 |
| Ch1 D-4 threshold CLI flags fail closed | Ch1 §17.3 | CHAPTER_1 §17.3 row 3 |
| STEP2 descriptive trace untracked | Ch2 §12.1 | docs/STEP2_BIDIRECTIONAL_SIEVE_DESCRIPTIVE_TRACE.md is in the clone |
| No executable gate covers Chapter 2 | Ch2 §14.3 | tests/test_chapter2_content_gate.py:1-8; 12/12 per CL0822A (chapter text not updated) |
| `_pump_deferred` docstring debt | FF §2.51 #2 | FF :2560-2562 (discharged at d8b21e3) |
| First production observation of both `_bp` falsifier fields | FF §2.51 #3; FF §2.54 | TB_RULING_BRIEF_I_PRODUCTION_SHAPE_FAILURE.md dispositions ("NOW LEGITIMATELY RECORDED", 30 and 126); the skill still says owed |
| S172 Run-4 Route-A routing patch | TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY | CL0828 §0–§5; tests/test_s172_run4_routing_patch.py; tests/run4_prelaunch_provenance_check.py |
| Hardcoded 0.25 miner threshold / Optuna thresholds dropped / staging controls swallowed | FF §2.7 #1, #2, #7 | FF :400-401, :407 |

## 7. Sources read, and surfaces unavailable

### Named sources (line counts)

| File | Lines | How read |
|---|---:|---|
| docs/TFM_PROJECT_FACTS_SKILL.md | 3,227 | §0.6, §0.7, §1 (all), §2.5, §2.7, §2.9, §2.17, §2.19 (head), §2.20, §2.27 (head), §2.29, §2.32 (head), §2.34 (defect paragraph), §2.50, §2.51, §2.54, §3, §8, header |
| docs/BACKLOG.md | 630 | in full |
| docs/PROJECT_FILE_CATALOG.md | 1,085 | §3.9, §7, §8, section index |
| docs/LEADS.md | 277 | in full |
| docs/TODO_MASTER_S101 / S114 / S120 / S122 / S127 / S132 / S135 / S139 / S143 / S145 / S148 / S150 / S152 / S154 / S163 / S170 | 102 / 172 / 169 / 182 / 196 / 130 / 143 / 130 / 159 / 211 / 172 / 76 / 103 / 135 / 132 / 136 | each in full |
| TODO_MASTER_S147.md, TODO_MASTER_S159.md (repo root) | 109 / 144 | each in full |
| docs/TODO_MASTER_S125b / S126 / S163KARG | 173 / 192 / 136 | by full diff against the neighbouring file already read in full (S122→S125b, S125b→S126, S126→S127, S163→S163KARG) |
| docs/TODO_SELFPLAY_AND_LLM_AUTONOMY.md | 425 | in full |
| docs/TODO_DISPLAY_AND_VISUALIZATION.md | 150 | in full |
| docs/TODO_PHASE7_WATCHER_INTEGRATION_REVISED_v3.md | 345 | status lines throughout; :236-345 in full |
| docs/TODO_PHASE7_WATCHER_INTEGRATION_REVISED.md | 206 | status lines throughout |
| INTEGRATION_ROADMAP.txt | 422 | :1-30, :240-295, :360-422, heading scan of the rest |
| docs/CHAPTER_1_WINDOW_OPTIMIZER.md | 2,310 | §17.3 |
| docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md | 1,502 | §12, §12.1, §14.3 |

### Other open-item lists found and consolidated

| File | Lines | What was taken |
|---|---:|---|
| docs/SESSION_CHANGELOG_20260828_RUN4_ROUTING_PATCH.md | 333 | in full; §7 "Carried forward" |
| docs/SESSION_CHANGELOG_20260822_WINDOW_ANCHOR_BRIEF_I.md | 117 | in full |
| docs/SESSION_CHANGELOG_20260822_BRIEF_I_PRODUCTION_SHAPE_REPAIR.md | 122 | in full |
| docs/SESSION_CHANGELOG_20260820_FIELD6_AND_WINDOW_ANCHOR_DESIGN.md | 145 | §5 |
| docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md | 153 | :60-153 |
| docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md | 214 | :1-80 |
| docs/TB_RULING_BRIEF_I_PATCH_REVIEW.md | 180 | :110-180 |
| docs/TB_RULING_BRIEF_I_PRODUCTION_SHAPE_FAILURE.md | 262 | dispositions table |
| docs/TB_RULING_WINDOW_ANCHOR_SEQUENCING.md | 66 | dispositions table and first two ruling items |
| docs/S172_WINDOW_ANCHOR_BRIEF_I_CLOSURE_STEP23_REPORT.md | 228 | §4 |
| docs/S172_BRIEF_I_PRODUCTION_SHAPE_CLASSIFICATION_REPORT.md | 374 | :305-335 |
| docs/S172_RUN4_PROPOSAL_PINNED_GEOMETRY.md | 197 | :1-60 |
| docs/RIG_CRASH_FORENSIC_20260822.md | 475 | :1-45 |
| docs/TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md | 292 | §4–§6, §7.2–§15 (the binding landing order P0-A/B/C, P1) |
| docs/PIPELINE_BEHAVIOUR_MODEL.md | 1,617 | :620-700 |
| docs/CHAPTER_14_TRAINING_DIAGNOSTICS.md | 3,199 | implementation checklist :3006-3081 |
| docs/CHAPTER_13_IMPLEMENTATION_PROGRESS_v3_9.md | 258 | :49-57, :128-147 |
| docs/SESSION_CHANGELOG_20260719_TELEGRAM_PROXMOX.md | 79 | :1-40 |
| agent_manifests/window_optimizer.json | — | parsed; `default_params`, `parameter_bounds`, :212-273 |

Code read for status checks (each cited by line in the tables): `agents/watcher_agent.py`, `agents/contexts/bundle_factory.py`, `agents/step_runner/command_builder.py`, `meta_prediction_optimizer_anti_overfit.py`, `train_single_trial.py`, `inner_episode_trainer.py`, `policy_conditioned_episode.py`, `chapter_13_orchestrator.py`, `chapter_13_acceptance.py`, `chapter_13_llm_advisor.py`, `diagnostics_analysis_schema.py`, `survivor_scorer.py`, `full_scoring_worker.py`, `generate_step3_scoring_jobs.py`, `run_step3_full_scoring.sh`, `run_scorer_meta_optimizer.sh`, `convert_survivors_to_binary.py`, `window_optimizer.py`, `window_optimizer_bayesian.py`, `window_optimizer_integration_final.py`, `coordinator.py`, `sieve_filter.py`, `sieve_gpu_worker.py`, `persistent_worker_coordinator.py`, `miner/*.py` (import lists; `range_miner_worker.py` :177-196, :328-340), `models/wrappers/xgboost_wrapper.py`, `tests/test_chapter2_content_gate.py` (header), `tests/test_s172_phase5_d5_process_sharded.py` (:2107-2119), `.gitignore`.

### Open-item lists found and not itemised

A grep for headings containing "open items", "carried forward", "what remains open", "deferred", "next steps", "known gaps" or "TODO" returned about 80 non-changelog files. These were not itemised, for the reason given:

- `docs/PIPELINE_BEHAVIOUR_MODEL.md` §16 (DIVERGENT register, 19 rows) and §17 (INCOMPLETE list). These are registers of doc-versus-code disagreement, recorded "neither adjudicated". They are a separate consolidation job.
- `docs/CHAPTER_13_IMPLEMENTATION_PROGRESS_v1_2 … v3_8`, `docs/S1xx_CHAT_PROMPT.md` (13 files), `S148_CHAT_PROMPT.md`, `S162_CHAT_PROMPT.md`, `CURRENT_STATUS.txt`, `CURRENT_Status.txt`, `SESSION_81_HANDOFF.md`, `SESSION_NOTES_20260102.md`, `SOAK_TEST_HANDOFF_PROMPT.md`: dated snapshots superseded by a later version that was read, or by the TODO_MASTER file of the same session.
- `docs/CLAUDE_CODE_REPORT_*`, `docs/TEAM_ALPHA_REVIEW_S172_PHASE4_REV*`, `docs/TB_SUBMISSION_GATE12_FORENSICS_F1_DEFECT.md`, `docs/CLAUDE_CODE_REPORT_F1_F2_R1.md` and the other Gate-12-era reports: their open items were either closed by the Gate-12 pass or carried into FF §2.51, which was read.
- `docs/PHASE7_PREREQUISITES.md`, `docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE_7_SOAK.md`, `docs/D6_THRESHOLD_AUTONOMY_SIGNPOSTS.md`, `docs/FALLBACK_PARITY_PASS1_20260815.md`, `docs/TEAM_ALPHA_6P2_TRANSITION_RULING_REQUEST.md`, `docs/CHAPTER_3_ALIGNMENT_AUDIT.md`, `docs/PROPOSAL_WATCHER_KPI_GOVERNANCE_STATES_v1_1.md`: not opened. Their subjects are represented above through the fact file, catalog or backlog, but anything they add beyond those is not in this report.
- The 181 session changelogs other than the five listed: not read.

### Rules and reminders in the sources that are not tasks (not counted)

| Where | What |
|---|---|
| BL §10 | standing reminders (VIR-6, gitignored files, keyword hits, cited is not read) |
| BL §17 | a skill revision lives in three places — the process rule; the owed revision is counted in group (c) |
| BL §18 | no Chapter-13 retrain approval or learning-loop invocation while a soak runs |
| BL §22 | `_operator_pin_params` / `_pin_bundle` are privileged seams; no new populator without governance review |
| Ch2 §12.1 | 44-entry registry vs 4 compiled variants — registry size is load-bearing for the uint8 encoding |
| FF §2.7 #3 | PWC hybrid threshold quarantined (`PWC_HYBRID_THRESHOLD_CONTRACT_UNCERTIFIED`), deliberately not repaired |
| CL0822A / CL0822B | Gate 22 blind spot (recorded, no change proposed); nested-tally rule; EXEC-PIN-1 standing rule |
| TM (all) | the "Architecture invariants" blocks at the foot of each TODO_MASTER file |

### Surfaces unavailable from this clone

- **Git history.** The public mirror has a single commit. No commit hash cited by the documents (`e9ca800`, `d8b21e3`, `48a8705`, `3e2537b`, …) could be checked; they are quoted from the documents.
- **Gitignored files.** `.gitignore:41` is `*.json`. Seven manifests and `watcher_policies.json` are force-tracked and were read. `agent_manifests/definitions.json`, `distributed_config.json`, all run outputs, sidecars, and the lottery datasets are absent.
- **Host state.** systemd units, cron, rig state, `~/dashboard_work`, the miner staging directory and ledger, `logs/`, the accumulator generations, installed skill copies.
- **The private repo and ser8.** Including the ~20 ser8-only changelogs.
- **Team Beta ruling texts** that were never saved as `TB_RULING_*` files.
- **Nothing was executed.** No test suite was run; test results ("26/26", "12/12", "0/9") are quoted from the documents named.

### Observations that are not list entries

- `docs/PROJECT_FILE_CATALOG.md` was regenerated on 2026-08-20. It describes BACKLOG as 19 entries, does not index `docs/LEADS.md`, and its gaps 10 and 11 have since closed.
- `docs/BACKLOG.md` §9 says Chapter 1 §17.3 has 14 open items; the table at this commit has 15 rows.
- Two TODO_MASTER entries stayed on the list for weeks after the code was fixed (CSV writer; variable-skip count). The series' checkboxes are not a reliable status, in either direction.
