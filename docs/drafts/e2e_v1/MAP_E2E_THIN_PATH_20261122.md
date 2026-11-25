# MAP: one thin path from run H to scored pools and a first selfplay episode (2026-11-22)

**Plain summary**
1. **No hit rate has been measured yet (0 of 0 pools graded).** No code today scores pools against unseen draws relative to `k/1000`. `evaluate_pools.py` grades one draw per call. `build_pools.py` and `backtest_pools.py` handle only xorshift32 on the legacy path. Chapter 13's grader cannot read Step 6's output (§2.5). The instrument was proposed as WP-3 (`docs/PROPOSAL_FINDINGS_20261113_RESOLUTION_v1_1.md` §2 WP-3, with rules in `docs/CLAUDE_CODE_INSTRUCTIONS_BINDING_ADDENDUM_BETA_R1_20261113.md` §F) and was never built. The 2026-11-14 handoff lists it as "needed before agents work on Steps 2 to 6" (`docs/TFM_SESSION_HANDOFF_20261114.md:72`). Status, not finding.
2. **Run H has nothing past Step 1.** Its tree has no `survivors_with_scores.json`, no `reinforcement_engine_config.json`, no sidecar and no pool. The only Step 2 output is the tracked 2026-03-11 `optimal_scorer_config.json` (read now).
3. **Everything from Step 3 to selfplay can run on VM101 alone, in an isolated copy, with no rigs.**
   - Step 3: GPU is optional, torch falls back to CPU.
   - Step 4: CPU.
   - Step 5: one RTX 3080 Ti for NN, XGBoost and CatBoost.
   - Step 6: CPU.
   - Selfplay inner episode: CPU.
   - None of it can go through WATCHER or the shell launchers without writing into the main tree or contacting the rigs (§3, the "(5) Cannot be isolated" entries in §2).
4. **The thin path is mostly harnesses, not code changes.** Three run-scoped in-process harnesses (Step 3, Step 6 vote, walk-forward grader) plus direct CLI calls (Step 4, Step 5 per family, selfplay) reach (a) and (b) without touching a governed or hand-landed file. The grader is new code and should go to Beta before its first counted result (WP-3 says so).
5. **The holdout as built is not clean for learned pools.**
   - The sieve never saw the holdout. Its window is 12 midday draws at offset 25 (`optimal_window_config.json`), far from the 80/20 tail (`window_optimizer.py:1170-1175`).
   - But Step 3 builds Step 5's target `holdout_quality` from `holdout_history.json`, so a Step 5-ranked pool graded on the same draws breaks S176 Phase C (`docs/TB_RULING_S176_WATCHER_KPI.md:692-725`).
   - The clean route is a run-scoped split: label block and grade block, given to Step 3 as a shorter `--holdout-history` file. No code change. Owner question Q1.
6. **Selfplay needs only Step 3's output.** One `--single-episode` run on run H is possible with no code change once Step 3 has run. The `neural_net` guard is not on that path, because the tree-only defaults never trip it (`inner_episode_trainer.py:72`, `:497-503`). What it yields:
   - per-model R², fitness and top features;
   - an indication of *what the trees use*, the owner's stated aim;
   - no hit-rate evidence.
7. **The recorded blockers that shape the numbers stay open.** These rows need governed decisions; the thin path measures under them and does not fix them, and the TODOs in §6 record them:
   - skip and position semantics (WP-1, BACKLOG §8.1, Ch7 N1/O3);
   - forward/reverse matches not merged into Step 3 (Option A/B);
   - run H's variable-skip vehicles regenerated as constant java_lcg at skip 0;
   - selfplay's target and features differ from Step 5's.
8. **Nothing is removed.** All 12,156 rows go through every step. No filter is added and no row is dropped. Every rule the path adds is a declared, configurable parameter.

**Owner questions**
- **Q1, the grade block.** For pools ranked by Step 5, split `holdout_history.json` (3,614 draws) into a label block for Step 3 and a later grade block that nothing upstream sees. Where should the split fall (suggestion: first half label, second half grade)? Should grading be midday-only (the sieve's stream, 1,808 holdout draws, about half in the grade block) or both sessions? An unlearned arm (Step 1 vehicles, no Step 5 ranking) can be graded on the whole holdout cleanly.
- **Q2, the vote position.** The baseline is Step 6's own convention: output index = combined record index, skip = the survivor's skip, which is 0 for every run H row (`prediction_generator.py:839-846`; NPZ `skip_min` 0 on 12,156 of 12,156). Which alternative arms, if any, should be declared before the first grade? Candidates:
  - a midday-stream index anchored at the sieve offset 25 [exact law UNVERIFIED, Ch7 O3];
  - one arm per skip in [0, 16].
  Each extra arm costs statistical power (§5).
- **Q3, go for runs.** The Step 3 harness, Step 4, Step 5 and the selfplay episode are pipeline steps, run on VM101 only. Each needs your go (CLAUDE.md §1.3). Do you want one go for the whole sequence or one per step?
- **Q4, selfplay features.** Selfplay trains on all feature-dict keys, including `score` and `confidence`, with target `holdout_hits`. Step 5 uses 89 features with `holdout_quality`. Should the first episode run as-is, or should a configurable target and exclude list be added first, default unchanged, so a second arm can mirror Step 5?
- **Q5, Step 2 on the critical path.** Step 3 reads only `prng_type` and `mod` from `optimal_scorer_config.json`, and both are absent, so the defaults apply (Ch4 B5). May Step 3 run on the existing tracked config while the O1 wiring proceeds separately, with this recorded as TODO-E2E-2?

- **Author and scope:** Claude (Team Alpha, agent 4), worktree `/home/michael/tfm_worktrees/e2e-map`, branch `e2e-map` at `41ad509`, uncommitted. Read-only research. Nothing was run except imports, `ls`, and read-only NPZ/JSON inspection. No GPU work, no rigs. Nothing was written outside this file.
- **Method:** three read-only research subagents covered (i) Steps 3–4, (ii) Steps 5–6 and the instrument, and (iii) Chapter 13 and selfplay. I spot-checked their key anchors myself:
  - `inner_episode_trainer.py:68-74`, `:495-503`;
  - `selfplay_orchestrator.py:905-935`;
  - `scripts_coordinator.py:495-502`;
  - `prediction_generator.py:836-850`;
  - `window_optimizer.py:1165-1176`;
  - `TB_RULING_S176_WATCHER_KPI.md:692-725`;
  - run H NPZ columns, `optimal_window_config.json`, `forward_survivors.json`.
  Other anchors are as the subagents read them at `41ad509` and are not re-opened here. Treat them as cited, not re-verified.
- **Binding direction:**
  - `OWNER_DIRECTION_e2e_20261122.txt` (20:37, 20:39, 20:46): thin, but done correctly; configurable; reversible; shortcuts recorded as TODOs.
  - The baseline ruling of 14:42: 0.30/0.30 is a moveable parameter.
  - O1 (wire Step 2 into Steps 3 and 5) and O2 (objective stays).
- **Yardstick:** seeds are vehicles. No statement here judges whether any seed is "found". The only verdict this path can produce is a pool hit count against `k/1000` on draws nothing upstream saw.

## Prior record

**Searches run** (worktree at `41ad509`, `/bin/grep -rlniE` unless noted):

| # | Pattern | Scope | Read |
|---|---|---|---|
| P1 | `end[- ]to[- ]end\|thin path\|e2e` | `docs/PROJECT_FILE_CATALOG.md`, `docs/TB_RULING_*`, `docs/TB_RULING_REQUEST_*`, `docs/PROPOSAL_*`, `docs/CHAPTER_*`, `docs/*_AUDIT.md`, `docs/TODO_CONSOLIDATED.md`, `docs/BACKLOG.md`, `docs/LEADS.md`, `docs/exchange/` | file list only. No document plans a Step 1→6 + grading + selfplay path on one generation |
| P2 | `hit rate on (real\|unseen\|held)\|never been measured\|k/1000`, then `hit.?rate` | same | `PROPOSAL_FINDINGS_20261113_RESOLUTION_v1_1.md` §1, §2 WP-1/WP-2/WP-3, §3; `PITFALL_STATE_RECOVERY_YARDSTICK.md` §5; `PROPOSAL_STEP1_SEARCH_RECORD_AND_RERUN_v1_1.md:265-267` (hit-rate record format) |
| P3 | `WP-3\|hit-rate instrument` | `docs/*.md` | `CLAUDE_CODE_INSTRUCTIONS_BINDING_ADDENDUM_BETA_R1_20261113.md` §F; `TFM_SESSION_HANDOFF_20261114.md:72`; `CLAUDE_CODE_REPORT_CPU_GPU_GENERATOR_CONTRACT_STAGE_A.md:205`; `exchange/from_beta/X-FINDINGS-PLAN_R2_BETA_REVIEW.md:16`, `:73-87` |
| P4 | `selfplay\|step 3\|step 5\|step 6\|pool` | `docs/TODO_CONSOLIDATED.md` | `:105-125` table (b), `:256-257`, `:275`, `:337`, `:352-355` |
| P5 | `first selfplay\|selfplay episode\|run selfplay` | as P1 | `CHAPTER_14_AUDIT.md`, `TODO_CONSOLIDATED.md` |
| P6 | headings | `docs/LEADS.md`, `docs/BACKLOG.md` | L-1…L-7; BACKLOG §1, §5, §8, §12, §15, §16, §18 |
| P7 | `find ~/tfm_worktrees -maxdepth 4 -iname '*hit_rate*'…` and `ls scripts \| grep -iE 'hit\|pool\|grade'` | all worktrees | **no instrument exists** in any worktree |
| P8 | subagent (i) | `CHAPTER_4_AUDIT.md`, `CHAPTER_5_AUDIT.md`, system map §7.1–7.3, §13, §14, BACKLOG §8.1, `CHAPTER_13_AUDIT.md` B2, WP-1 Stage A Q3–Q6, `TB_RULING_*` (30 files, `-l`) | as reported |
| P9 | subagent (ii) | `CHAPTER_6_AUDIT.md`, `CHAPTER_7_AUDIT.md`, system map §10, `TB_RULING_S179_IMPLEMENTATION_AUTH.md` §2.5, §2.6, §6, `TB_RULING_S176_WATCHER_KPI.md` Phase C, WP-2 Stage A report, `drafts/SKIP_DESIGN_RECORD_20261121.md` §4.4 | as reported |
| P10 | subagent (iii) | `CHAPTER_13_AUDIT.md` (B1–B14, A.0), `CHAPTER_14_AUDIT.md`, `TFM_SELFPLAY_NN_STATUS_20261119.md` (in full), `TODO_SELFPLAY_AND_LLM_AUTONOMY.md`, system map §4, §10–§14, `TB_RULING_S179_IMPLEMENTATION_AUTH.md`, fact file §2.1 | as reported |

**Documents that already decide, schedule or record the subject:**
- **WP-3 hit-rate instrument.** Its rules are fixed by the documents below, and it was never built.
  - `PROPOSAL_FINDINGS_20261113_RESOLUTION_v1_1.md` §2.
  - Measurement rules (Beta R2 §4, quoted in §1 there): "never saw" means unseen by the sieve **and** by training, target construction, Optuna selection and threshold tuning.
  - Grader rules: addendum §F.
- **WP-1 Stage A (skip/position contract).** Report `29a6cb9`. Stage B awaits owner sign-off. BACKLOG §8.1.
- **WP-2 Stage A (Step 6 → Chapter 13 grading contract).** Report `ea85436`. Q8 options A/B/C proposed, not implemented.
- **S176 Phase C:** causal walk-forward (`TB_RULING_S176_WATCHER_KPI.md:692-725`).
- **S179 §2.5 / §6:** unique pool ranking and draw identity, approved, not implemented.
- **Step 2:** `~/tfm_worktrees/pwc-offset/docs/drafts/PLAN_STEP2_REPAIR_20261122.md`, `NOTE_STEP2_READINESS_ON_RUNH_20261122.md`, Beta `X-STEP2-WIRING_R1`. **Not redone here.**
- **System map** §7.3 (F-PARITY, forward/reverse Option A/B), §4.2 (attribution blockers, A-NORM), §12.4 (NN-Q / NN-A).
- **Chapter audits 4, 5, 6, 7, 13, 14.** The B/S/N/O items are cited per row in §3.
- **TODO_CONSOLIDATED (b):** the selfplay NN guard and the y-normalisation port at `:117` (REVIEW); NN y-normalisation in Step 5 is done (`:337`, `train_single_trial.py:504-510`, `:689-690`).

**Labels:**

| Item raised here | Label |
|---|---|
| No hit-rate instrument exists | **ALREADY RECORDED** (WP-3; handoff `:72`) |
| Holdout used for Step 3's target, so a learned pool graded on it leaks | **ALREADY RECORDED** as a rule (Beta R2 §4; S176 Phase C). **NEW** as a statement about run H's split, together with the label/grade-block route |
| Hardcoded main-tree output path `scripts_coordinator.py:499-500` | **NEW** (no record found, subagents i and iii) |
| Step 3 localhost jobs read main-tree `scoring_chunks/` (stale Feb 2026) and write main-tree `full_scoring_results/` | **NEW** |
| `/dev/shm/prng/step3` reused by file name | **NEW** for Step 3 (the Step 2 note records the same check, also unrecorded) |
| WATCHER P0.5 fleet preflight fires for local Steps 4 and 5 through the `lottery_data` key | **NEW** [by reading] |
| Stale `models/reinforcement/best_model.txt` in run H can be mis-credited by the compare archive | **NEW** [by reading] |
| `holdout_quality` built at skip 0 for every run H row (no per-seed skip) | **NEW** as a run H fact. The mechanism is **ALREADY RECORDED** (Ch6 S2; WP-1 Stage A Q4) |
| Everything else in §3 | **ALREADY RECORDED**, cited per row |
| **CHANGES A RULING** | **none.** Q1–Q5 are owner decisions inside existing rulings |

**Not searched:**
- the host main tree's runtime state, beyond `ls` of run H;
- the ser8 archives, the private repository, project files outside the repository;
- the `apply_s*.py` and changelog corpora;
- `docs/exchange/` contents beyond Beta X-STEP2-WIRING R1 and the X-FINDINGS-PLAN R2 hits;
- `SELFPLAY_ARCHITECTURE_PROPOSAL_v1_0.md` and `SELFPLAY_INTEGRATION_PROGRESS_v1_0.md` bodies;
- the rig trees.

## 1. Run H inputs as they stand (read now)

| Item | Value | Source |
|---|---|---|
| Generation | 12,156 rows, 22 arrays, all `prng_type` 1 (java_lcg_hybrid), `skip_mode` 1, `skip_min` 0, `skip_max` 16 on every row (trial-level, not per seed) | `~/skipsw_run_thr3030/bidirectional_survivors_binary.npz` (numpy read) |
| `forward_matches` / `reverse_matches` | [0.333, 0.667] / [0.333, 0.583] | same |
| Step 1 config | window 12, offset 25, sessions `['midday']`, `executed_thresholds` 0.30/0.30, one trial | `optimal_window_config.json` |
| Histories | `train_history.json` 14,454 and `holdout_history.json` 3,614, the combined 80/20 split of 18,068 (bare int lists, not session-separated) | `window_optimizer.py:1170-1175` |
| Holdout dates | 2021-03-17 → 2026-02-26; 1,808 midday and 1,806 evening | subagent (ii), computed from the dataset [not re-run here] |
| Sieve window | midday [25, 37) = 2002-11-29 → 2002-12-10 | subagent (ii) [not re-run here] |
| Step 2 output | `optimal_scorer_config.json`, tracked, byte-identical to main (2026-03-11) | Step 2 plan summary item 3 |
| Step 6 survivor inputs | `forward_survivors.json` is a stub `{"survivor_count": 2368586, "note": …}` | read now |
| Stale artifact | `models/reinforcement/best_model.txt` (14:31) | `ls` now |

## 2. Per step: command, hosts, blockers, route, what cannot be isolated

Row order in each blocker table is the order a run would hit them. "Route" means the smallest change or run-scoped diagnostic that gets past the row. **HL** = owner-hand-landed (`scripts/owner_land.py:86-87`: `agents/watcher_agent.py`, `agent_manifests/`, `miner/`, `gate12_launch.sh`, `scripts/gate12_parity_gate.py`). **GOV** = in `GOVERNED_FILES` (`scripts/gate12_parity_gate.py:174-200`, includes `prng_registry.py`).

### 2.1 Step 3: Full Scoring

**(1) Command WATCHER would run** (`STEP_SCRIPTS[3]`, `agents/watcher_agent.py:409`; params `:2012-2026`, named style `:2361`):

`bash run_step3_full_scoring.sh --survivors bidirectional_survivors_binary.npz --config optimal_scorer_config.json --train-history train_history.json --holdout-history holdout_history.json --chunk-size 1000 --jobs-file scoring_jobs.json --prng-type java_lcg --mod 1000 --batch-size 100`

- The last four flags are ignored (`run_step3_full_scoring.sh:120-123`).
- The child gets no `cwd=` (`watcher_agent.py:2894-2903`).
- The chain:
  1. ramdisk preload (`.sh:33-40`);
  2. `generate_step3_scoring_jobs.py` (`:160-178`);
  3. scp (`:203-215`);
  4. `scripts_coordinator.py` (`:226`);
  5. pull (`:248-258`);
  6. aggregate into `survivors_with_scores.json` (`:299-408`);
  7. floor check `<46` features (`:475-478`).
- The worker CLI is `full_scoring_worker.py --seeds-file --train-history --holdout-history --gpu-id --output-file` (`:522-549`).

**(2) Hosts.**
- **Through the launcher:** localhost plus the rigs at bare-metal IPs (`distributed_config.json`).
- **GPU optional.** Torch runs on `cuda` if present, else on CPU (`survivor_scorer.py:117`). The java_lcg torch generator on CPU equals `_generate_sequence` (WP-1 Stage A Q4).
- **VM101 alone is feasible for 12,156 rows:** 13 chunks of 1,000.
- **[UNVERIFIED] estimate:** 5–30 min, peak RSS about 2–3 GiB per worker process. The chunk prediction tensor is 1,000 × 14,454 int64 ≈ 116 MB. The first run must measure and record both (CLAUDE.md rule 9).

**(3) Recorded blockers, in bite order:**

| # | Blocker | Anchor | Status |
|---|---|---|---|
| 1 | Preflight SSH hard-fails: bare-metal IPs are down while the rigs boot Proxmox | `watcher_agent.py:1866-1875`; `preflight_check.py:317`, `:346-354`; Ch4 audit Q6 | **ruled** not a bug (CLAUDE.md §3); `rig_profile` **deferred** |
| 2 | `optimal_scorer_config.json` must exist (`ramdisk_preload.sh:92-99`, `set -e`) | `preflight_check.py:285` | exists. Only `prng_type`/`mod` are read and both are absent, so the defaults apply. Ch4 **B5 open** (O1 wiring) |
| 3 | Remote jobs fail over SSH and are retried locally; the ramdisk ssh has no ConnectTimeout | `scripts_coordinator.py:739-741`, `:1068-1112`; `ramdisk_preload.sh:49`, `:131` | same record as #1 |
| 4 | **Main-tree coupling.** Localhost jobs `cd` to the main tree (`scripts_coordinator.py:893`, `:992`). Output path hardcoded (`:499-500`). `REMOTE_BASE` (`.sh:83`) and the generator default `remote_base_path` (`generate_step3_scoring_jobs.py:155`, `:315`) also point at the main tree. Localhost would read stale Feb-2026 `scoring_chunks/` and the run H aggregator would exit 1 (`.sh:238-242`) | lines given | **NEW**, open |
| 5 | `/dev/shm/prng/step3` "already loaded" is by file name only | `ramdisk_preload.sh:63-76`, `:110-112` | **NEW**, open |
| 6 | Sequential fallback merges 6 of 18 metadata fields; F-PARITY does not exist | Ch4 B6; system map §7.3 item 2; TODO `:62`, `:161` | P0-B **authorised, not implemented** |
| 7 | Hybrid variable-skip vehicles are regenerated as constant `java_lcg` at skip 0 | `full_scoring_worker.py:344`; Ch4 audit §7.4; Ch3 audit F8 | recorded, **open** |
| 8 | Holdout labels: `holdout_hits` at raw `T+j`; `holdout_quality` (the Step 5 target) at raw `skip_min + j` = `0…3613` on run H, overlapping the training-feature positions | Ch4 S1; BACKLOG §8.1; TODO `:164`; WP-1 Stage A Q4 | S1 **switched off by ruling**; WP-1 Stage B **awaits owner**; fix lives in **GOV** `prng_registry.py` |
| 9 | Session/anchor: the sieve ran midday at anchor 25; Step 3 assumes combined record r ↔ raw output r | WP-1 Stage A Q4/Q6; BACKLOG §5 | **open** (WP-1) |
| 10 | `forward_matches`/`reverse_matches` carried and not merged | Ch4 S2; system map §7.3 item 3 | TB: do not add silently. Option A/B **open** |
| 11 | 5 dead placeholders; `<46` floor against 91 | Ch4 B7; BACKLOG §15 | **open**, not a stop |

Rows 7–10 decide what Step 5 learns to rank on. They do not stop the run. As a hit-rate prediction: if the ranking signal is tied to stream position, these rows lower the learned pool's hit rate. The grader in §2.7 measures that against the unlearned arm on the same draws.

**(4) Route.**
- **Harness H3:** a run-scoped, in-process script. Run it with cwd set to a copy of the run H inputs under `data/e2e_runH/` in this worktree.
  - Load: `utils/survivor_loader.load_survivors` on the resolved NPZ, then the chunker from `generate_step3_scoring_jobs.py` (in memory).
  - Score: `full_scoring_worker.score_survivors(...)`. Run `inspect.signature` first (CLAUDE.md §1.7). Smoke one chunk before all 13.
  - Aggregate: the `.sh`'s inline aggregation, reproduced in Python. **TODO-E2E-3:** duplicated logic, record the anchor `.sh:299-408` and test parity on the smoke chunk.
  - Rows 1, 3, 4 and 5 vanish. Code changes: none.
  - Optional: run the sequential fallback on a 500-row subsample and diff the 18 metadata fields. That is a run-scoped F-PARITY *diagnostic*, not the gate.
  - The `holdout_history` argument is the Q1 **label block** file, not the full holdout.
- **The landable fix for row 4**, so the launcher can run from any tree, is three small changes. `scripts_coordinator.py` is coordinator code, so it is in Beta's before-commit class.
  - Make `scripts_coordinator.py:499-500` use the node's `script_path`.
  - Pass `--remote-base-path` through in the `.sh`.
  - Give `distributed_config` a run-scoped `script_path`.
  - **Not needed for the thin path.** Recorded as TODO-E2E-4.

**(5) Cannot be isolated:** the launcher → coordinator → `distributed_worker` chain (main-tree paths), the host-wide `/dev/shm/prng/step3`, and the rig trees. The harness avoids all three.

### 2.2 Step 4: Adaptive Meta-Optimizer

**(1) Command** (`watcher_agent.py:412`; `ml_meta.json`):
`python3 adaptive_meta_optimizer.py --window-results optimal_window_config.json --lottery-data train_history.json --mode full --apply`
- Writes `reinforcement_engine_config.json` and `optimization_results/meta_optimization_*.json` to cwd (`adaptive_meta_optimizer.py:1001-1081`).

**(2) Hosts:** local CPU, seconds, under 1 GiB [UNVERIFIED].

**(3) Blockers:**

| # | Blocker | Anchor | Status |
|---|---|---|---|
| 1 | Preflight SSH, as Step 3 #1 | — | as above |
| 2 | Preflight wants `survivors_with_scores.json`, which Step 4 never opens | `preflight_check.py:286`; Ch5 D10 | recorded |
| 3 | `lottery_data` key triggers P0.5 dataset freeze plus `fleet_preflight` | `watcher_agent.py:492`, `:2075-2155` | **NEW** [by reading]; whether the targets include rigs is [UNVERIFIED] |
| 4 | Flat `optimal_window_config.json` (no `best_result`/`all_results`) → constants | Ch5 S1 | **ruled** TB Q4, preserve the drift |
| 5 | Constant architecture, entropy discarded, feedback has no producer | Ch5 B1–B4 | **open** |
| 6 | Step 5 never reads Step 4's output | Ch5 B5; Ch6 B7 | **open** |

**(4) Route:** run the CLI directly in the `data/e2e_runH/` copy. No code change. **It is off the critical path** (row 6). Run it for completeness and record its output as unconsumed (TODO-E2E-5).

**(5) Cannot be isolated:** only WATCHER's preflight and P0.5, which the direct call avoids.

### 2.3 Step 5: Anti-Overfit Training, all four families including the NN

**(1) Command** (`watcher_agent.py:411`; `reinforcement.json:173-184`; flags `:2410-2428`):
`python3 meta_prediction_optimizer_anti_overfit.py --survivors survivors_with_scores.json --lottery-data train_history.json --compare-models --trials 20 --output-dir models/reinforcement --enable-diagnostics --model-type catboost --enable-vmap --k-folds 20`
- `--compare-models` runs one child per family, in the order neural_net, lightgbm, xgboost, catboost (`meta_prediction_optimizer_anti_overfit.py:148`, `:226-269`, `:3112-3130`).
- The children lose `--k-folds`, `--output-dir` and `--enable-vmap` (Ch6 **B2**).
- The winner is copied to `models/reinforcement/` (`:313-321`). Optuna storage is cwd-relative `optuna_studies/step5_*.db` (`:1744-1766`), unlike Step 1's hardcoded live path.

**(2) Hosts:** VM101 alone.
- NN through `train_single_trial.py` on CUDA.
- XGBoost CUDA, CatBoost GPU, LightGBM CPU (`:265`).
- Imports verified in `~/venvs/torch`: xgboost 3.0.4, lightgbm 4.6.0, catboost 1.2.8, torch 2.5.1+cu121, optuna 4.4.0 (subagent ii).
- X is 12,156 × 89 ≈ 4 MB. Peak RSS under 2 GiB [UNVERIFIED]; runtime [UNVERIFIED].

**(3) Blockers:**

| # | Blocker | Anchor | Status |
|---|---|---|---|
| 1 | No `survivors_with_scores.json` in run H | `ls` | status: needs Step 3 |
| 2 | P0.5 fleet preflight through WATCHER (`lottery_data`) | `watcher_agent.py:492`, `:2083-2135` | **NEW** [by reading] |
| 3 | Children drop `--k-folds` / `--output-dir` | Ch6 B2 | **open** |
| 4 | Stale `best_model.*` in `models/reinforcement/` can be archived under a failed child | `meta_prediction_optimizer_anti_overfit.py:274-278`; run H has `best_model.txt` | **NEW** [by reading] |
| 5 | Target `holdout_quality` built at skip 0 and from holdout draws, so leakage (summary item 5) | `holdout_quality.py:26-51`; `:1481` | Ch6 S2 recorded; leakage handled by Q1 |
| 6 | Signal gate fires only on a constant target (exit 2, Step 6 skips) | `:405-412`, `:651-660` | as built |
| 7 | Test split never evaluated (B1); final NN uses default 256-128-64, not Optuna's best (B3); CatBoost iterations/depth ignored (B4); Step 4 output unread (B7); holdout offset unrecorded (B8) | Ch6 audit | **open**, none fatal |
| 8 | Refit defect and zero-filled missing feature names | Beta X-STEP2-WIRING R1 §2–§3 | **open** |

**(4) Route:**
- Run four direct calls in the `data/e2e_runH/` copy, one per family, `--model-type <f> --trials 20 --k-folds 20`, with no `--compare-models`. This passes rows 2, 3 and 4. Start from an empty `models/reinforcement/` in the copy (the run H tree is not touched).
- Pick the winner by a declared rule, written into the run record before the runs.
- **TODO-E2E-6:** the manual compare replaces `_s88_run_compare_models`. Fix B2 in the module (not governed, not hand-landed) and return to `--compare-models`.
- No code change for the thin path.

**(5) Cannot be isolated:** nothing, provided the copy holds `train_single_trial.py` in cwd. Otherwise `subprocess_trial_coordinator.py:199-202` falls back to the main tree's worker. **TODO-E2E-7:** check this in the smoke run.

### 2.4 Step 6: Prediction pools

**(1) Command** (`prediction.json` `default_params`):
`python3 prediction_generator.py --config reinforcement_engine_config.json --survivors-forward forward_survivors.json --survivors-reverse reverse_survivors.json --lottery-history <P0.5-frozen daily3> --k 20 --k 20 --models-dir models/reinforcement`
- `pool_size` maps to `--k` (Ch7 Q2).
- Writes `predictions/next_draw_prediction.json` plus `predictions/history/`, cwd-relative (`prediction_generator.py:905`).
- It produces **one pool of up to 20 integers for the one draw after the history**, repeats kept (`:836-850`).

**(2) Hosts:** CPU, VM101.

**(3) Blockers:**

| # | Blocker | Anchor | Status |
|---|---|---|---|
| 1 | Needs the Step 5 sidecar; fatal without it | `prediction_generator.py:290-294` | status |
| 2 | `forward_survivors.json`/`reverse_survivors.json` are stubs; `int("survivor_count")` raises `ValueError` | `survivor_scorer.py:136`; `:1044` | Ch7 **B1**, no ruling |
| 3 | The dual-sieve path coerces survivors to ints, recomputes features at default settings and votes at skip 0; never reads Step 3's rows | Ch7 Q2, **N1** | **open**, not ruled |
| 4 | Vote position = combined index `len(history)`, generator phase 0 at combined index 0; the sieve was midday at anchor 25 | `:839`; Ch7 **O3**; WP-1 Stage A §1.5 | **open** |
| 5 | No schema-hash check; missing columns → 0.0 | `:798`; Ch6 B5 | **open** |
| 6 | Missing checkpoint gives a mean-of-features ranking reported as success | Ch7 **B2** | **open** |
| 7 | Repeats kept in the pool | Ch7 **B3**; S179 §2.5 | approved, **not implemented** |
| 8 | NN scaler applied only when the checkpoint carries it; otherwise silently skipped | `:385-450`, `:814-824` | as built |
| 9 | **The single pool targets a draw that is not in the dataset** (index 18,068, after 2026-02-26). It cannot be graded from the tree | `:839`; no later draws found in the tree [host/ser8 not searched] | **NEW** as a thin-path fact |

**(4) Route: Harness H6, the run-scoped vote.**
- Load `PredictionGenerator` once with the Step 5 sidecar.
- Feed it the survivor dicts from Step 3's `survivors_with_scores.json`, with `use_dual_sieve=False`. This passes rows 2 and 3, uses the population Step 5 trained on, and keeps every row.
- For each grade-block draw t, take the ranking once (it does not depend on t) and vote with Step 6's own rule at `next_idx = t`.
- Collapse repeats only for counting distinct `k` (§2.7). The raw pool is kept as written.
- **TODO-E2E-8:** H6 calls the module's ranking and vote code; it does not reimplement them. Anchor `:777-850`. Test that at t = `len(history)` it reproduces the CLI's own pool byte for byte on the same inputs.
- **TODO-E2E-9:** the dual-sieve path (B1/N1) is left in place and untouched.

**(5) Cannot be isolated:** nothing. It is cwd-relative.

### 2.5 Chapter 13

**(1) Command:** `python3 chapter_13_orchestrator.py --once --no-llm`.
- `--once` needs no `new_draw.flag` (`chapter_13_orchestrator.py:1086-1087`).
- It reads from cwd: `prediction_pool.json`, `confidence_map.json`, `lottery_history.json` (`{"draws": […]}`, last draw only), `best_model.meta.json`, `survivors_with_scores.json`, `watcher_policies.json` (`chapter_13_diagnostics.py:50-55`, `:132-170`).
- It writes the S140b annotation into `prng_analysis.db` (`:292-319`).
- A standalone `chapter_13_diagnostics.py --generate --predictions --history --survivors --model-meta --output` exists (`:1004-1030`).

**(2) Hosts:** CPU, VM101.

**(3) Blockers:**

| # | Blocker | Anchor | Status |
|---|---|---|---|
| 1 | Steps 3/5/6 outputs absent | — | status |
| 2 | No `lottery_history.json` → outcome `error` | Ch13 B2 | **open** |
| 3 | The grader calls `.get()` on Step 6's integer list; a missing pool is graded as 0 predictions (233 of 233 historical cycles had `pool_size 0`) | `chapter_13_diagnostics.py:243`; Ch13 B1, A.0.1, A.0.3; WP-2 Stage A | WP-2 Stage B **not implemented** |
| 4 | `ranked_predictions.json` has no writer. It is read only in the hit-regression branch, which never fired | `:325-345`; system map §4.2 | **governed**, attribution P0-C; off this path |
| 5 | Triggers structurally invalid; approvals run nothing; cooldown never reset | Ch13 B3, B5, B7; `TB_RULING_S176` | **open** |
| 6 | `best_rank` counts ±5 near-misses as hit@20 | WP-2 Stage A §1.4, §1.8 | **open** |
| 7 | `HALT_FLAG=/tmp/agent_halt` is box-global | `chapter_13_orchestrator.py:74` | recorded by subagent iii; record [not found] |

**(4) Route.** Chapter 13 is **not on the critical path to (a)**. Its grader would give a wrong count (rows 3 and 6). The WP-3 grader in §2.7 measures hit rate instead.
- **After (a), one optional observe-only smoke:** run the standalone diagnostics with explicit paths. Inputs:
  - an operator-written `lottery_history.json` (train + label block, in `draws` form);
  - a dict-shaped `prediction_pool.json` adapter from H6's pool.
- **TODO-E2E-10:** the adapter stands in for the WP-2 contract. Remove the adapter once WP-2 Stage B lands.

**(5) Cannot be isolated:** `/tmp/agent_halt` (box-global). Everything else is cwd-relative.

### 2.6 Selfplay

**(1) Command.**
- **WATCHER route:** `agents/watcher_agent.py --dispatch-selfplay` (`:3871`, `:4199-4202`) → `dispatch_selfplay` (`agents/watcher_dispatch.py:116`).
  - It stops the LLM (`:156-158`).
  - It runs `selfplay_orchestrator.py --survivors survivors_with_scores.json --episodes 5 --policy-conditioned` with `cwd` = **code root** (`:170-175`, `:195`).
  - It has a 1 h timeout (`:193`).
- **Direct route:** `python3 selfplay_orchestrator.py --config configs/selfplay_config.json --survivors survivors_with_scores.json --single-episode --policy-conditioned --no-emit-candidate --n-jobs 6 --project-root .` (`selfplay_orchestrator.py:1101-1167`).
- **One inner episode:**
  1. Load the survivor dicts (`:870-893`).
  2. Apply the policy. It is empty if `learned_policy_active.json` is absent (`policy_conditioned_episode.py:53`, `:140-154`).
  3. Build X from the whole `features` dict, sorted keys, with no exclusion (`selfplay_orchestrator.py:911-927`, read now). Set y = `holdout_hits`, falling back to `score` (`:933`, read now).
  4. Fit LightGBM, XGBoost and CatBoost, each with 3-fold CV and 100 estimators, on CPU (`inner_episode_trainer.py:72`, `:186-227`, `:381-400`).
  5. Score fitness `max(0, R²−0.05) − 0.2·stability − 0.3·overfit` (`:136-159`).
- **Outputs:** `telemetry/learning_health*.json[l]`, `learned_policy_candidate.json` (suppressed by `--no-emit-candidate`), `policy_history/`, stdout.

**(2) Hosts:** CPU only. The outer GPU episode is off (`use_coordinator: false`, `:194-196`, `:597-607`). VM101 has 8 cores; the config's `n_jobs: 22` oversubscribes, so pass 6. Peak RSS under 2 GiB [UNVERIFIED].

**(3) Blockers:**

| # | Blocker | Anchor | Status |
|---|---|---|---|
| 1 | No `survivors_with_scores.json` in run H | — | status: needs Step 3 (only Step 3) |
| 2 | **The `neural_net` guard** raises `ValueError` if `model_types` names `neural_net` or `random_forest`. The tree-only default never trips it, so it is **not on the first-episode path**. `MODEL_BUILDERS` has no NN entry, so removing the guard alone fails with "Unknown model type" (`:228-232`, `:265-267`) | `inner_episode_trainer.py:71`, `:497-503` (read now) | TODO `:117` **REVIEW**; system map §12.4 gates **NN-Q and NN-A**, owner rules |
| 3 | Target and features differ from Step 5: `holdout_hits` and all keys including `score`/`confidence`, against `holdout_quality` and 89 features | `:911-933`; `meta_prediction_optimizer_anti_overfit.py:612`, `:629`, `:1481-1482`; Ch14 audit `:84`; fact file `:416-420` | **open** ("Open: selfplay inner loop still uses R²") |
| 4 | Acceptance needs `val_r2 ≥ 0.80`; the recorded real-data R² is 0.000155, so no candidate can be accepted | Ch13 B13 | **governed, open**. Irrelevant with `--no-emit-candidate` |
| 5 | Episodes identical (same file, same policy, fixed seed); `propose_transform_update` is a no-op | Ch13 B14; `policy_conditioned_episode.py:373-382` | B14 **open**; no-op **switched off by ruling** |
| 6 | The promotion seam drops transforms; nothing requests selfplay | Ch13 B11, B10 | **open**, off this path |
| 7 | S179 dispatch governance gates not in code | `TB_RULING_S179_IMPLEMENTATION_AUTH.md:79-110` | **authorised, not built**. WATCHER route only |

**(4) Route.**
- **First episode:** the direct CLI in the `data/e2e_runH/` copy, after Step 3, `nohup … | tee` into `logs/`. No code change.
- **Q4 option:** add a configurable target name and exclude list to `SelfplayConfig` and `_survivors_to_xy`, default unchanged, so a second arm mirrors Step 5's 89 features and `holdout_quality`. The file is not governed and not hand-landed; the TODO row says REVIEW, so send it to Beta.
- **The NN port, for a later episode:**
  - port `train_single_trial.py:train_neural_net` (`:450`) pieces: device pick `:473-480`, X scaler `:486-492`, y-normalisation `:497-510` with its inverse `:689-690` and checkpoint `y_mean`/`y_std` `:718-719`, `:778-779`;
  - add an NN builder;
  - remove the guard in the same change;
  - run it in subprocess isolation (`TODO_MASTER_S145.md:82-86`; `DESIGN_INVARIANT_GPU_ISOLATION.md`);
  - behind NN-Q/NN-A unless the owner rules otherwise; Beta.
  - Size: about 1 day. **Not on the first-episode path.**

**(5) Cannot be isolated:** the WATCHER route (code-root cwd). The direct CLI is fully cwd-scoped: no absolute paths, no Optuna.

### 2.7 The hit-rate instrument (WP-3): does it exist?

**No.** P7 found nothing, and subagent (ii) confirmed it by reading:

| Tool | What it does | For run H |
|---|---|---|
| `evaluate_pools.py:28`, `:36-40` | one pool, one draw (date+session). Baseline `k = int(label)/1000` uses the pool's label, not its distinct count | unusable as a walk-forward |
| `build_pools.py:66-78`, `:170`, `:210-224` | reads `results/multi_gpu_analysis_*.json`; xorshift32 only | cannot read run H |
| `backtest_pools.py:56-120` | legacy `residue_sieve` per date → build → evaluate | legacy path |
| Chapter 13 grader | §2.5 rows 3, 6 | wrong count |
| `full_scoring_worker.py:274-308` | per-vehicle holdout exact-hit rate | a label, not a pool |

**Route: Grader G, a new run-scoped CPU script under `scripts/diagnostics/`.** It touches no governed or hand-landed file. Its rules are taken whole from WP-3, addendum §F, Beta R2 §4 and S176 Phase C, and written into its header before any run:
- **Inputs:** H6's per-draw pools, the frozen dataset (sha256 bound), the grade-block definition, the arm list, the abstention rule.
- **Per draw t:** `k_t` = the number of **distinct** integers in [0, 999] (validate the range and type). One membership hit per draw. Record the draw identity (date, session, combined index), the pool sha256, the configuration identity, and every abstention with its denominator.
- **Errors, not misses:** missing, stale or duplicate inputs.
- **Statistic:** expected hits Σ `k_t`/1000. This is a Poisson-binomial with a one-sided test, reported with an interval. Correct for the number of arms declared in advance.
- **Controls:**
  - uniform-random pools of the same `k_t`, which must land inside a declared tolerance of chance;
  - the most frequent `k` values in the train block (draw-frequency control);
  - a fixture with a known hit count;
  - the addendum §E controls: duplicates, out-of-range values, a draw of `000`, and pending vs miss.
- **Outputs** under `data/e2e_runH/grade/`, per-draw JSONL plus a summary with sha256.

**Arms for the first grade** (declared before the run):
- **A0, unlearned:** run H's 12,156 vehicles ranked by a declared Step-1-only key, e.g. seed order or `forward_matches + reverse_matches`, voting with Step 6's rule. Clean on the **whole** holdout.
- **A1, learned:** Step 5 winner via H6, graded on the **grade block** only (Q1).
- **R, random control.** Q2 may add position arms.

**Power** (normal approximation, k = 20, p = 0.02):

| Grade draws n | Expected hits | SD | Extra hits for one-sided α 0.05 at 80% power (≈2.5 SD) |
|---|---|---|---|
| 3,614 (all, A0 only) | 72.3 | 8.4 | ≈ 21 (lift ≈ 1.29) |
| 1,808 (midday, A0) | 36.2 | 5.95 | ≈ 15 (lift ≈ 1.41) |
| ≈ 904 (midday, half holdout, A1) | 18.1 | 4.2 | ≈ 11 (lift ≈ 1.58) |

**Size:**
- 6–10 h of agent work including the fixtures, then Beta.
- Run time: minutes.
- Memory: 12,156 vehicles × ≤ 3,614 positions, streamed, small.

**TODO-E2E-11:** G is the WP-3 instrument's first form. Its header rules must match WP-3 and §F verbatim, so that it can become the committed instrument rather than a second one.

## 3. Shortcuts taken for speed (explicit TODOs)

| TODO | Shortcut | Anchor to revisit |
|---|---|---|
| E2E-1 | Run in a copy under `data/e2e_runH/` in this worktree, not through WATCHER | `agents/watcher_agent.py:2894-2903` (no `cwd`); §2.1 (5) |
| E2E-2 | Step 3 runs on the tracked 2026-03-11 scorer config; Step 2's O1 wiring continues separately | Ch4 B5; `PLAN_STEP2_REPAIR_20261122.md` O1 |
| E2E-3 | H3 reproduces the `.sh` aggregation in Python | `run_step3_full_scoring.sh:299-408` |
| E2E-4 | Main-tree path hardcodes left in place | `scripts_coordinator.py:499-500`; `.sh:83`; `generate_step3_scoring_jobs.py:155` |
| E2E-5 | Step 4 run but unconsumed | Ch5 B5; Ch6 B7 |
| E2E-6 | Manual per-family compare instead of `--compare-models` | Ch6 B2; `meta_prediction_optimizer_anti_overfit.py:226-269` |
| E2E-7 | Worker fallback path not exercised | `subprocess_trial_coordinator.py:199-202` |
| E2E-8 | H6 calls the module's ranking and vote; must reproduce the CLI's pool | `prediction_generator.py:777-850` |
| E2E-9 | Dual-sieve Step 6 path left as is | Ch7 B1, N1 |
| E2E-10 | Dict-pool adapter for the Chapter 13 smoke | WP-2 Stage B |
| E2E-11 | Grader G to become the WP-3 instrument | WP-3; addendum §F |
| E2E-12 | Vote position and skip are Step 6's convention (combined index, skip 0) and are not settled | WP-1 Stage B; Ch7 O3; BACKLOG §8.1 |
| E2E-13 | Label/grade split is a run-scoped history file, not a configured pipeline split | `window_optimizer.py:1170-1175`; BACKLOG §5 |

Every value the path introduces is a CLI argument or a run-record field, never a constant in code: grade split, arms, `k`, the ranking key, the winner rule, `n_jobs`, sample size. Nothing is deleted.

## 4. Recommended shortest route

Each step: work, rough size, GPU or not, and owner gate.

**To (a), first pools scored against `k/1000` on unseen draws:**
1. **Owner answers Q1, Q2, Q3 and Q5.** Write them as `OWNER_*` lines.
2. **Build the run copy and record.** `data/e2e_runH/` gets the resolved NPZ (sha256 checked against generation `3f6a884d…`), train history, the label-block and grade-block history files, the configs, and a run record with every declared parameter. About 1 h, no GPU.
3. **H3, Step 3 harness.** Smoke one chunk, then all 13, plus the optional F-PARITY diff. Agent time: 3–5 h. Run time: 5–30 min [UNVERIFIED]. GPU optional. **Owner go.**
4. **Step 4, direct.** Minutes, no GPU. Can run alongside step 5.
5. **Step 5, four families direct.** Agent time about 2 h. VM101 GPU. Run time [UNVERIFIED]; record it. **Owner go.**
6. **H6, the vote harness.** 3–4 h of agent time, which can be built in parallel against a fixture sidecar. No GPU.
7. **G, the grader**, built in parallel from now, with its fixtures and the uniform-random control passing. 6–10 h, no GPU. **Beta before the first counted result.**
8. **Grade arms A0, A1 and R.** Minutes. Write the result record with null results first and every count with its denominator.

**To (b), the first selfplay episode:** steps 1–3, then:

9. **The selfplay direct CLI**, `--single-episode --no-emit-candidate --n-jobs 6`. Under 1 h of agent time; CPU; minutes [UNVERIFIED]. **Owner go.** Option: the Q4 configurable target/exclude arm first, about 2–3 h plus Beta.

**Parallel split, separate worktrees:**
- Agent A: steps 2, 3, 4, 5 and 9 (the data chain; sequential, needs owner gos).
- Agent B: step 7, the grader (no data dependency).
- Agent C: step 6, H6 (needs only the sidecar format until step 5 lands).
- Agent D, optional: the Q4 selfplay arm, then the NN port behind NN-Q/NN-A.

The Step 2 wiring agent (O1) already has its own worktree (`~/tfm_worktrees/step2-repair`) and is not on this path.

**Critical path:** about 1.5–2 agent-days of wall clock to (a). About half a day to (b), because it needs only steps 1–3.

**What each result would and would not tell us:**
- **(a), arm A0:** whether run H's sieve vehicles, voting by Step 6's rule, hit more often than `k/1000` on 3,614 draws the sieve never saw.
  - It does not say why, does not test other thresholds or skip rules, and does not test the learning steps.
- **(a), arm A1:** whether Step 5's ranking adds hits over A0 on the same grade block.
  - Under the open rows of §2.1 (7–10) and §2.4 (3–4) it is a measurement of *this configuration*, not of the design.
  - **A null result for either arm closes no lead** (lead-handling skill). It is a measured null for that declared configuration, reported first with its denominator. The follow-ups are the declared alternative arms (Q2) and the WP-1 contract.
- **(b):** per-model R², fitness and the top features that three tree models use to fit `holdout_hits` on run H. That is a first indication of what carries signal in the 91 features, which is the owner's stated aim.
  - It gives no hit-rate evidence.
  - With `score` and `confidence` among the features (row 3), their importance must be reported as such, not as a discovery.
  - It cannot propose anything, because `propose_transform_update` is a no-op.
  - Repeated episodes would be identical (B14).
