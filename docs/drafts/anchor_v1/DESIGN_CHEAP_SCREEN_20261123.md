# DESIGN · Step 1 cheap screen: a per-trial score from the trial's own pools (DRAFT, NOT LANDED)

Status: DRAFT for the lead. Code in the worktree only; nothing committed, landed, launched or run on a rig or GPU. (VM101 was switched off at 20:25; nothing for this task had been written by then; restarted from the reading.)
Task: `logs/e2e_session/TASK_CHEAP_SCREEN.txt` (lead, 2026-11-23 20:21). Worktree `anchor-brief2`, base `41ad509`.

## Plain summary

**What it is.** A setting, `--step1-objective`, with three values:
- `bidirectional_count`, the default: survivor count, today's objective, unchanged;
- `screen_hits`;
- `screen_rank`.

With a screen value, each Step 1 trial is scored by its own survivors, unlearned and ranked as run H's A0 arm. They vote by Step 6's own rule on a 600-draw **selection block** that starts right after that trial's window. The score is against the pinned grader's exact null: hits at k = 20 as −log10 p, or rank of the drawn number as −z. Every screen number is DEVELOPMENT evidence.

**What each trial may see.**
- Its window, and the draw values of its own selection block, through one guarded accessor.
- Nothing at or after the block's end.
- Nothing in a confirmation block. Every selection block must end by the Step 1 holdout split (combined index 14454) and before every confirmation block. The anchor domain is narrowed so that every sampled window fits, never clamped.

So S176 Phase C holds for any draw from 14454 on. The walk-forward then grades only `[14454, 18068)`: `--range-start 14454`, with the used selection blocks declared as previously inspected.

**Built on Brief II.** The screen requires the derived anchor domain and the holdout guard. At window 12, midday anchors go up to 6395 (guard alone: 6695); Run I's anchor cannot be screened.

**Tests** (CPU fixture). Final run 3: **15/15 gates, 13/13 mutants detected.** The Brief II suite is still **18/18 and 12/12** against base 41ad509. The tests show:
- the default study is identical to 41ad509, in both parameters and trial attributes;
- a window whose survivors' pools hit its selection block wins the real Optuna search, where survivor count cannot tell the windows apart;
- a selection block overlapping a confirmation block is refused;
- no draw outside the block is read;
- the pool equals Step 6's own function, with ties.

None of this says whether a screen score predicts later hits; that is what the screen makes cheap to measure.

**Cost** (measured at run H's shape, synthetic draws). Run H's trial took 130.8 s.
- `screen_hits` adds about **0.3 s** per trial, plus about 2 s once per run for an import.
- `screen_rank` over all 12,156 survivors adds **18-140 s** single-core, or 78 s on 2 workers at the latest window. Capped at 1,000 survivors it adds 2-12 s.

**Throughput.** About **650 windows per day** at run H's seed count (2²⁸) with `screen_hits`. Thousands need several days, or a smaller seed count per trial [UNVERIFIED scaling]: screen cheaply, then give the few best the full treatment.

**Resume.** The study is tagged with the objective and the settings' sha256. Relaunch `--resume-study` in chunks of about 50 trials under nohup; a changed setting or objective is refused.

**Multiplicity.** The best of 1,000 null windows reaches −log10 p ≈ 3.2 by chance. Confirm one declared winner once, with m = 1: k = 20 needs about 1,550 draws for a 1.5× lift, which `[14454, 18068)` holds (development). Counted confirmation needs draws after 2026-02-26, which are not in the dataset yet.

**Hand-landed entries.** Manifest and registry entries are a draft diff only. The manifest default is `""` so default argv is unchanged.

**Owner questions:**
- Q1: may the objective change for screen runs?
- Q2: the Beta system-map rule against outcome-chosen windows;
- Q3: should agents be able to propose the objective?

**Found on the way:**
- HEAD became a WIP snapshot (`b78a15c`) mid-task. Both suites now prove identity against 41ad509, not HEAD.
- Variable-skip records carry `java_lcg_hybrid`. The screen now votes them as Step 6 does instead of refusing them.

## Prior record

### Searches (2026-11-23 20:40, main tree `~/distributed_prng_analysis` at 41ad509, read only)

```
P='cheap screen\|screen_rank\|screen_hits\|step1-objective\|step1_objective\|a0_hits_selection\|selection block'
grep -il "$P" docs/PROJECT_FILE_CATALOG.md docs/TB_RULING_* docs/TB_RULING_REQUEST_* docs/PROPOSAL_* \
     docs/CHAPTER_* docs/*_AUDIT.md docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/LEADS.md docs/exchange/*/*
   -> 0 files on every surface
grep -ril "$P" ~/tfm_worktrees/*/docs/drafts
   -> ch14-learn/docs/drafts/ch14_v1/PLAN_CH14_LEARNING_LOOP.md (F15, LL-17, Q5)
      selfplay-nn/docs/drafts/loop_v1/READY_LOOP_SELFPLAY_20261123.md:339 (Q3: selfplay needs its own selection block)
P2='window selection\|objective.*hit\|hit.*objective\|Optuna objective'   (same main-tree surfaces)
   -> TB_RULING_S176_WATCHER_KPI.md:724; PROPOSAL_STEP1_SEARCH_RECORD_AND_RERUN_v1_0.md:47;
      PROPOSAL_EPISTEMIC_AUTONOMY_UNIFIED_v1_3.md:84; CHAPTER_13_AUDIT.md:142 (-> system map §10); CHAPTER_6_*  (read: Step 5 objective, not Step 1)
```

### Documents that decide, authorise, schedule or record the subject

| Document | What it says | Label |
|---|---|---|
| `TASK_CHEAP_SCREEN.txt`; `OWNER_GO_cheap_screen_20261123.txt` (owner 20:17, 20:20 "Sure") | Owner asks for the screen first, so that thousands of windows are worth running | the request |
| `PLAN_CH14_LEARNING_LOOP.md` F15 (:116), LL-17 (:147), Q5 (:255) (worktree ch14-learn) | Step 1 objective is survivor count; LL-17b plans `--step1-objective {bidirectional_count, a0_hits_selection}`, scored on a selection block the grade block never reuses; Q5 asks the owner whether Optuna's objective should change at all | **ALREADY RECORDED** (plan). Owner 20:20 "Sure" answers the lead's "moved up?"; Q5 itself is not answered in those words. See owner Q1 |
| `docs/TB_RULING_S176_WATCHER_KPI.md` Phase C :692-727 | No data from draw t or later may enter training, feature extraction, **window selection**, survivor scoring, threshold selection or prediction generation for draw t | **ALREADY RECORDED**; binds the design (§2) |
| `docs/TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md` §10 :214-218 | Beta: Step 3's `holdout_hits` is a training label only, "forbidden as a filter, weight, mask, **window** or production-time feature" | **CHANGES A RULING (in spirit).** The screen does not read `holdout_hits`. But it does choose a window by outcome hits on a pre-holdout block. Owner rulings 2026-11-23 10:40 (Beta holds no longer valid) and 18:57 (a Beta rule is not binding unless the lead says so) apply. The owner's word is asked (Q2) |
| `OWNER_RULING_selfplay_contract_not_binding_20261123.txt`, lead position | Kept as a leak control: "a candidate is never judged on the draws it is later tested on" | **ALREADY RECORDED**; it is the selection/confirmation disjointness rule here |
| `docs/PROPOSAL_STEP1_SEARCH_RECORD_AND_RERUN_v1_0.md:47` | Reruns become autonomous only after a walk-forward baseline exists | **ALREADY RECORDED**; unchanged (the screen is operator-launched) |
| S115 M2 pruning, `window_optimizer_integration_final.py:2420-2429`; `window_optimizer_bayesian.py:416`, `:776` | A trial with forward_count == 0 is pruned before the reverse sieve | **ALREADY RECORDED**; the screen keeps it, the AlphaGo "do not finish a path that already looks like a failure" |
| AlphaGo note `claude/NOTE_ALPHAGO_LOOP_FOR_SELFPLAY_20261121.md` | Not in the repository (`docs/TFM_SESSION_HANDOFF_20261121.md:178`). Not read; the lead's summary in the task file is used | **NOT READ** |
| Brief II (this worktree): `utils/window_anchor_domain.py` (sha `a9fc243d…`), `READY_ANCHOR_BRIEF2.md` | Derived anchor domain and holdout guard, both default off, unlanded | **ALREADY RECORDED (unlanded)**; the screen requires both |
| H6 vote harness `~/tfm_worktrees/e2e-vote/scripts/diagnostics/e2e_vote_harness.py`; `READY_LOOP_MISS_20261123.md` §0.1, §1.1 | Step 6 vote rule; A0 = `field_sum(forward_matches+reverse_matches)`, tie `seed_asc`, k 20 (run H `full_A0/run_record.json`); full vote matrix 12,156 × 3,614 in 54.7 s on 6 workers | **ALREADY RECORDED (unlanded)**; the screen uses the same rule |
| Grader `~/tfm_worktrees/e2e-grader/scripts/diagnostics/e2e_grader.py` sha `c7fe48bf…` (development pin, TASK_LOOP_ADDENDUM_3) | R4 k_t distinct, R7 expected hits, R8 exact Poisson-binomial tail, R22 development mode, R24 rank | **ALREADY RECORDED (unlanded)**; called, not copied |
| WF driver `~/tfm_worktrees/e2e-map/scripts/diagnostics/e2e_walk_forward.py` sha `8597d939…` | Blocks over `[range_start, range_end)` by date; `--previously-inspected START:END` recorded per arm | **ALREADY RECORDED (unlanded)**; the disjointness interface (§3) |

Not searched: the host outside the listed worktrees, ser8 archives, the private repository, and project files that are not in the repository (including the AlphaGo note).

## 1. The screen score

One number per trial, computed on VM101 right after the trial's sieve, from that trial alone.

| Part | Rule | Anchor |
|---|---|---|
| Survivors | The trial's own bidirectional records: the canonical records of its miner assembly, constant then variable, exactly as `ingest_assembly` appends them | `miner/step1_ingress.py:219-220`; `window_optimizer_integration_final.py` `_build_test_result_from_miner(keep_screen_records=…)` |
| Ranking (unlearned) | Run H's A0 key: `forward_match_rate + reverse_match_rate`, each cast to float32 first (NPZ `forward_matches`/`reverse_matches` hold these values as float32), ties broken `seed_asc` | `e2e-vote/data/e2e_runH_vote/full_A0/run_record.json` (`field_sum`, `seed_asc`, k 20); `e2e_vote_harness.py` `build_key` |
| Vote | Step 6's rule. For target t the top-k ranked survivors each vote `seq[t]` of their own sequence (`SurvivorScorer._generate_sequence`), skip by `PredictionGenerator._get_survivor_skip`, fallback `default_skip` (0; run H A0 used skip 0 for all 12,156) | `prediction_generator.py:836-850`, `:530-552`; `survivor_scorer.py:119-125` |
| Selection block | `[s0, s1)` in combined index. `s0` = combined index of the window's last draw + 1 + `selection_gap` (0). `s1 = s0 + selection_draws` (600, the WF block length). Graded rows: `selection_sessions` (`all` by default; `trial` or a list selectable) | `utils/step1_screen.py` `ScreenPlan.selection_block`, `graded_targets` |
| `screen_hits` | `-log10 P(X ≥ H)` under the grader's exact Poisson-binomial null: k_t = distinct pool values, p_t = k_t/1000, H = membership hits | grader R4, R5, R7, R8: `validate_pool_values`, `grade_draw`, `pb_upper_tail` (called, pinned by sha256) |
| `screen_rank` | `−z`, where z is the grader's normal-approximation z of the drawn numbers' mid-rank sum. The ordering is the full vote: value v scores −(rank of the first survivor voting v), and values nobody votes tie last. The null is exact per draw, from that draw's own mid-ranks. The grader's Monte-Carlo control must pass, or the trial is refused | grader R24: `draw_ranks`, `rank_summary`; ordering rule = `e2e_vote_harness.py` `value_scores_for` (`FULL_VOTE_RANK_BASIS`) |
| No survivors / pruned | Score 0 for both: empty pools give p = 1, and an all-tied ordering gives z = 0. A pruned trial (S115 M2: forward_count == 0) raises `TrialPruned` before scoring, as today | `window_optimizer_integration_final.py:2420-2429` |
| Label | Every number is **DEVELOPMENT** evidence (grader R22 wording). Never a counted result | `ScreenPlan.run_record` |

The score is stored as the Optuna trial value. Its detail is the trial user attribute `step1_screen`, which records: block, graded draws, survivors, voters, k, hits, expected hits, exact p, mean rank, z and seconds.

**Why both objectives.** Hits at k=20 see about 12 expected hits in 600 draws, so they are coarse. Rank uses every draw's full ordering, so it is the finer instrument. It is also the one the lead added for development (TASK_LOOP_ADDENDUM_1 item 2). Neither is a verdict on TFM. Each is a prediction about hit rate on later draws, which only the confirmation block (§7) can test.

## 2. What each trial may see (S176 Phase C)

Phase C (`docs/TB_RULING_S176_WATCHER_KPI.md:720-727`): nothing from draw t or later may enter window selection for draw t.

For a trial with window filtered `[a, a+w)` under session filter S, with last combined index `c_last`:

| Data | Seen by the trial | Enforced by |
|---|---|---|
| Residues of the window `[comb(a), c_last]` | yes (the sieve) | unchanged |
| Draw **values** of the selection block `[s0, s1)`, `s0 > c_last` | yes, to grade its own pools | `ScreenPlan.draw_value`: the only read of a draw value, refused outside the open block (gate G4, mutant M2) |
| Index t of each selection target | yes (Step 6 reads only `len(history)`, `:839`) | the vote takes t, not history |
| Any draw at or after `s1` | **no** | `draw_value` |
| Any confirmation block | **no**: `s1 ≤ min(selection_end_max, start of every confirmation block)`; overlap refused | `check_block`; plan load refuses a confirmation block starting before `selection_end_max` (G3) |
| Survivors of other trials | no (Optuna sees only their scores; that is the search) | — |

So a window chosen by the screen for forecasting draw t ≥ `selection_end_max` has seen nothing from t or later.

Default `selection_end_max` = the Step 1 holdout split, `step1_holdout_split_point(18068)` = 14454 (`window_optimizer.py`, Brief II). The default confirmation block is that holdout, `[14454, 18068)` (`confirmation_blocks: "holdout"`).

The anchor domain is narrowed by `AnchorDomain.anchor_max_hook` so that every sampled window's selection block fits. On the live dataset (18,068 rows), the maximum anchor at window 12 is:
- midday: **6395** (last window draw at combined 13852, 2020-05-20), against the guard-only maximum of 6695;
- both sessions: 13842.

A pinned anchor that does not fit is refused, never clamped: Run I's 6695 cannot be screened.

**Not a Phase C point but a leak to name.** The selection block lies in Step 3's train history: Step 3 computes features from `train_history.json` = `[:14454]`. That is allowed by Phase C (it is all before the confirmation draws). But it means the screen and Step 3 features both see the same pre-holdout draws.

## 3. Where the selection block comes from; disjointness from walk-forward and confirmation blocks

**For any anchor:**
- The selection block is the `selection_draws` combined rows right after the window (§1).
- It is a function of `(anchor, window_size, sessions)` only, so every trial has its own block. Different trials' blocks may overlap each other; that is the search and costs nothing, because all of them lie below `selection_end_max`.

**The WF driver** (`e2e_walk_forward.py` sha `8597d939…`) grades blocks over `[--range-start, --range-end)`, cut by date (`cut_blocks`, `:171-187`). A run that uses a window chosen by a screen study:
1. passes `--range-start <selection_end_max>` (14454 by default). Then every WF grade block is inside the confirmation region and disjoint from every selection block by construction;
2. passes the union of the selection blocks actually used, `[min s0, max s1)` read from the study's `step1_screen` trial attributes, as `--previously-inspected START:END`. The WF driver hands it to the grader as `previously_inspected_blocks` (`:1144`). In development mode the grader records an overlap; in counted mode R13 refuses one (`e2e_grader.py` R13, R22). With rule 1 the overlap is empty, so the declaration is a second wall, not the first.

Labels and Step 5 training at a WF origin may use selection-block draws, because they are before the origin. Grading may not.

**WF `--range-start auto`.** Under Addendum 1 this starts shortly after the sieve window. With a screen-chosen window that would grade selection blocks, so it must not be used. This is a rule for the operator and for the WF agent; the WF file is not edited here (TODO-SCREEN-3).

**Selfplay** validates on block i−1 (TASK_LOOP_ADDENDUM_3). Its blocks are WF blocks, so they are inside the confirmation region and disjoint from every screen selection block.

## 4. Cost per trial

Measured here (VM101, taskset cores 4,5, nice 15): one java_lcg `_generate_sequence` of 18,068 values takes **14.7 ms**, about **0.81 µs per value**. The mean is over 20 seeds; 700 values take 0.57 ms. From the MISS agent's run H matrix (12,156 × 3,614 out to index 18,067, 54.7 s on 6 workers), it is about **1.49 µs per value**, including fork and scheduling.

The vote needs each voter's sequence out to the block's last index, so its cost is about voters × s1 × (0.8 to 1.5 µs). That is the estimate. The **measured** probe is `data/anchor_v1_logs/screen_cost_probe.py` (sha `c273c530…`), log `screen_cost_probe_run2.log` (`890af562…`) plus the valid first four lines of `screen_cost_probe_run1_unguarded.log`. It ran at run H's shape: 12,156 survivors with 7 distinct key values, window 12 midday. The draws are synthetic (live dates and sessions, random values), so no real outcome was read. Cores 4,5, nice 15.

| Objective | Voters | Workers | Run H's window (anchor 25, block [1112, 1712)) | Latest midday window (anchor 6395, block [13853, 14453)) |
|---|---|---|---|---|
| `screen_hits`, k 20 | 20 | 1 | 2.37 s (first call in the process: includes the one-off `prediction_generator` import, about 2 s) | **0.32 s** |
| `screen_rank`, `rank_survivor_cap` 1,000 | 1,000 | 1 | 2.28 s | 12.38 s |
| `screen_rank`, all survivors | 12,156 | 1 | 18.35 s | 140.21 s |
| `screen_rank`, all survivors | 12,156 | 2 | 18.42 s | 77.69 s |

Peak RSS was 578 MiB.

At the early window, the uncapped rank cost is mostly per-draw work (600 orderings ranked, plus the grader's 2,000-rep Monte-Carlo control), not sequence generation; workers do not help there. At the late window, generation dominates (12,156 × 14,453 values), and 2 workers nearly halve it.

Every early-window `screen_hits` figure above 2 s (2.35-2.67 s, including the repeats in the run 1 log) is the first call in a fresh process. In the repeats, that process was a spawn child re-running the unguarded probe. The extra time is the `prediction_generator` import, which is paid once per Step 1 process.

**Comparison with a trial.** Run H's trial took **130.8 s** for 2²⁸ seeds, skip 0-16, both modes (RUNBOOK_RUN_I §5, receipt). `screen_hits` adds well under 1 %. `screen_rank` over all survivors can roughly double a late-anchor trial, unless it is capped or given workers. `workers > 1` uses a spawn pool, because the Step 1 process holds threads and sockets.

## 5. Windows per day

Assumptions:
- Run H's measured trial time (130.8 s, 2²⁸ seeds), sequential trials (`n_parallel` 1, which the derived domain requires);
- the 59.2 s publication paid once per run, not per trial [UNVERIFIED for a multi-trial run];
- the fleet up all day;
- no gaming on VM100: lead rule, NOTE_zeus_reboot 2026-11-23.

| Seeds per trial | Trial time | Windows per day |
|---|---|---|
| 2²⁸ (run H) | 131 s + screen | **about 650** with `screen_hits`; about 400-600 with `screen_rank` uncapped, depending on anchor |
| 2²⁶ | about 33 s if sieve time is linear in seeds [UNVERIFIED: fixed per-trial overhead not measured] | about 2,500 |

So "thousands of windows" means a two- to five-day run at run H's seed count, or a one-day run at a quarter of it.

Survivor counts fall with the seed count too, which makes each screen noisier. The AlphaGo-shaped plan:
1. a cheap screen (fewer seeds, `screen_hits`), with pruning as today (S115: a trial with no forward survivors stops before the reverse sieve);
2. the full treatment (full seed range, Step 3-6, walk-forward) for the top few windows only.

Limits not measured:
- rig throughput differs by 12-15 % (`.122` faster);
- sustained 8-GPU load took `.122` and `.121` down on 2026-08-19 (memory: rig-ct100-four-cores-eight-gpus);
- leads L-6 and L-7 (rig GPU faults) are open.

A multi-day run will meet rig faults, so §6 matters.

## 6. Resume after a rig fault or a host reboot

What exists:
- The Optuna study is SQLite under `optuna_studies/`. `--resume-study --study-name <name>` continues it (`window_optimizer_bayesian.py:682-720`).
- Completed trials keep their values and their `step1_screen` attributes.
- A trial running at a crash is left RUNNING or FAIL and earns no credit (STEP1-SEARCH S7).
- The coordinator does one retry, then fails the trial, on a stripe fault (S172 TB Q3).

What the screen adds:
1. The study is tagged `step1_objective = <objective>:<settings sha256>` at its first trial (`screen_study_check`). A resumed run with another objective or other screen settings is refused before its sieve, and so is a survivor-count run (G6). A screen run on a study that already holds untagged completed trials is refused too.
2. The plan is rebuilt from the same `--screen-config` file, the frozen dataset digest and the pinned grader on every launch. A changed file changes the tag, so the run is refused.
3. The screen is a pure function of the trial's records and the frozen dataset, so re-scoring is never needed after a resume.

**Operating shape** (a proposal, nothing launched):
- `nohup` launches of M trials each (`--trials M`, M ≈ 50, about 2 h), every launch with `--resume-study --study-name <same>`.
- After a rig fault: re-run the pre-flight (CT100s reachable, 8 cards each, parity) and relaunch the same line.
- After a host reboot: the same, plus check that the Proxmox boot state is restored (memory: rigs default-boot to Ubuntu; `boot-proxmox` is needed). Never reboot a host unasked.

Not covered:
- `--resume-checkpoint` (D6.2) is refused with the derived domain (Brief II), so a crash inside a trial loses that trial only.
- The Step 1 search ledger key for the objective (TODO-SCREEN-2).

## 7. Multiplicity and the confirmation block

**Inside the screen.** The best of N windows is inflated. Under the null, the best `screen_hits` value (−log10 p) of N independent windows has a median of about log10 N + 0.16 and a 95th percentile of about log10 N + 1.3 (continuous approximation; the exact binomial is coarser):

| N | median best | 95 % best |
|---|---|---|
| 10 | 1.17 | 2.29 |
| 100 | 2.16 | 3.29 |
| 1,000 | 3.16 | 4.29 |
| 5,000 | 3.86 | 4.99 |

A screen winner below these values is what chance alone gives. The screen ranks windows; it does not show anything.

**Confirmation.** One winner, declared before the confirmation block is graded, tested once, with m = 1. The selection is paid for by never grading the selection block again, not by dividing α by N. Keeping the top M for confirmation needs a Bonferroni (or Holm) correction over M, which the grader does (R10).

Draws needed (one-sided exact binomial, α 0.05, power 0.8, pool k, true lift λ):

| k | λ 1.25 | λ 1.5 | λ 2 | λ 3 |
|---|---|---|---|---|
| 20 | 5,450 | 1,550 | 500 | 150 |
| 100 | 1,050 | 300 | — | — |
| 300 | 275 | 75 | — | — |

The available confirmation regions:
1. The Step 1 holdout `[14454, 18068)`, 3,614 draws. These are DEVELOPMENT only, because they have been looked at (TASK_LOOP_COMMON). Enough for λ ≥ 1.5 at k 20.
2. Draws after 2026-02-26, which are **not yet in the dataset** (the last row is 2026-02-26 midday). Counted. About 730 a year, so a year of new draws confirms λ ≈ 2 at k 20, or λ ≈ 1.25 at k 100.
3. A new generation (Run I).

Recommendation: confirm the single best screen window at k 20, 100 and 300 on the walk-forward over `[14454, 18068)` (development), then on new draws (counted).

## 8. The setting, code and tests

### 8.1 The setting

`--step1-objective {bidirectional_count, screen_rank, screen_hits}`, default **`bidirectional_count`**: today's `BidirectionalCountScorer`, unchanged. Two more settings travel with it:
- `--screen-config <json>`: schema `tfm.step1_screen.v1`. Every key and its default is in `SCREEN_DEFAULTS`; unknown keys are refused. `grader_path` and `grader_sha256` are required.
- the Brief II settings `--derived-anchor-domain --anchor-holdout-guard`, which are required for a screen.

A screen is refused with: no config; a config without a screen objective; another strategy, `n_parallel` ≠ 1, `--resume-checkpoint`, `--config-file` or a non-miner route (inherited from Brief II); a grader with another sha256 or drifted signatures; `mode` other than `development`; a confirmation block starting before `selection_end_max`; a study tagged by another objective or screen setting.

### 8.2 Code (worktree only; nothing landed)

| File | Change | Default path |
|---|---|---|
| `utils/step1_screen.py` (new) | `ScreenPlan` (geometry, refusals, Phase C accessor, A0 ranking, Step 6 vote, grader calls), `build_screen_plan`, `load_grader` (sha256 pin plus `inspect.signature` of the five grader functions), `screen_study_check`, `value_scores_for`, `vote_matrix` (spawn pool when `workers` > 1) | not imported |
| `utils/window_anchor_domain.py` | `AnchorDomain.anchor_max_hook` (default None), applied in `effective_range`; `trial_record` shows `screen_anchor_max` when set | None, so unchanged (Brief II suite) |
| `window_optimizer.py` | `ScreenScorer(ScoringFunction)` beside `BidirectionalCountScorer`; CLI `--step1-objective`, `--screen-config` and their refusals; hop 2 (`run_bayesian_optimization(*, step1_objective, screen_config)`, forwarded only when not default); `optimal_window_config.json` gains `step1_screen` only when on | forwarded nothing |
| `window_optimizer_integration_final.py` | hop 3 `optimize_window(step1_objective, screen_config)`; plan resolved once beside the anchor domain (same dataset, frozen digest, split function); `dataclasses.replace(domain, anchor_max_hook=plan.anchor_max)`; scorer `BidirectionalCountScorer() if _s1_screen_plan is None else ScreenScorer(plan)`; `_build_test_result_from_miner(keep_screen_records=False)`, passed `True` only when `coordinator._step1_screen_on`; study tag check beside the anchor-domain one; run record `step1_screen` | scorer, call and records unchanged (G-DEFAULT-WIRING) |
| `window_optimizer_bayesian.py` | trial user attribute `step1_screen`, written only when the result carries a screen record | attrs identical to base (G-DEFAULT-OBJ) |
| `tests/test_step1_screen.py` (new) | 14 gates, 12 mutants | — |

Not touched:
- `miner/`, kernels, the worker, `agents/watcher_agent.py`;
- `agent_manifests/`, `agents/registry/` (§9 is a draft diff only);
- `distributed_config.json`, the NPZ, Steps 2-6.

### 8.3 Tests

`tests/test_step1_screen.py` runs on CPU on a fixture. The fixture has 2,000 combined rows; rows [500, 1100) are planted so that the draw at t is `seq[t]` of one seed, and the confirmation block is [1600, 2000). Run with `taskset -c 4,5 nice -n 15`, one process at a time.

| Gate | What it proves |
|---|---|
| G-PLANTED | A trial whose survivors' pools hit its selection block scores higher than one whose pools do not, under both objectives. An empty trial scores 0. The block for anchor 590, window 10 is [600, 700) |
| G-PLANTED-OPT | The real `OptunaBayesianSearch` with `ScreenScorer` and the hooked domain, 24 trials. Every trial reports the same survivor count, so the count objective cannot tell windows apart. The best trial's block lies in the planted rows and beats every other trial. Every completed trial carries `step1_screen` |
| G-DEFAULT-OBJ | Default objective: a seeded study's trial parameters **and** trial user-attr keys are identical to the base commit's `window_optimizer.py` and `window_optimizer_bayesian.py`, loaded from git |
| G-DEFAULT-WIRING | AST of `optimize_window`: one function-level `_s1_screen_plan = None`; the scorer expression is `BidirectionalCountScorer() if _s1_screen_plan is None else …`; records stay off unless `_step1_screen_on` |
| G-OVERLAP | A confirmation block starting before `selection_end_max` is refused at load. A selection block overlapping a confirmation block, or ending after its start, is refused |
| G-PHASEC | The block starts at window last + 1. Scoring reads exactly `s1 − s0` draw values, all inside the block (sealed list). `draw_value` outside the block, or with no block open, is refused |
| G-HOOK | For 3 session options × 3 window sizes, the hooked maximum = min(guard maximum, plan maximum) < guard maximum. The block fits at the maximum and not one past it; a pin one past it is refused "never clamped". With the hook off, nothing changes |
| G-DISTINCT | k_t counts distinct values (20 copies of one seed give k_t = 1) |
| G-POOL-IDENTITY | The screen's pool equals `PredictionGenerator._build_prediction_pool`'s output (the module's own function) at two targets, with tied keys ordered `seed_asc` |
| G-ORDERING | `value_scores_for` equals the vote harness's own function (`e2e_vote_harness.py` sha `bbb04e81…`) |
| G-GRADER-PIN | A wrong grader sha256 is refused |
| G-STUDY-ARM | A tagged study refuses a default run, another setting and another objective. An untagged study with trials refuses a screen run and is left untouched by a default run |
| G-CLI, G-ROUTE | CLI refusals; hop 2 and 3 signatures; the manifest is untouched |

Mutants (applied, executed via `_MUT_HIT`, detected at the credited assertion):

| Mutant | Change | Credited at |
|---|---|---|
| S1 | overlap check removed from `check_block` | G-OVERLAP [OVL-check-block] |
| S2 | load-time confirmation check removed | G-OVERLAP [OVL-load] |
| S3 | block starts at the window's last draw | G-PHASEC [PC-after-window] |
| S4 | `draw_value` guard removed | G-PHASEC [PC-draw-guard] |
| S5 | hook ignored in `effective_range` | G-HOOK [HOOK-max] |
| S6 | k_t = raw pool length | G-DISTINCT |
| S7 | tie order reversed | G-POOL-IDENTITY |
| S8 | scorer returns survivor count (ignores the objective) | G-PLANTED-OPT |
| S9 | grader pin skipped | G-GRADER-PIN |
| S10 | screen attr written on a default trial | G-DEFAULT-OBJ |
| S11 | scorer selection inverted | G-DEFAULT-WIRING |
| S12 | study tag mismatch accepted | G-STUDY-ARM |

RESULTS: see §8.4 (filled from the logs).

### 8.4 Results

Logs under `data/anchor_v1_logs/` (gitignored `data/`):

| Run | Code | Result | Wall time, peak RSS | Log sha256 |
|---|---|---|---|---|
| Screen suite run 1 | before the S4 gate fix and the base fix | 14/14 gates, **11/12 mutants**: S4 was caught by the wrong assertion (the sealed list fired before the guard check) | 61.1 s, 562 MB | `screen_suite_run1.log` |
| Screen suite run 2 | base = 41ad509; S4 fixed | 14/14 gates, 12/12 mutants, PASS | 55.6 s, 557 MB | `0421527f…` |
| Screen suite **run 3** (final code) | plus `_hybrid` handling, G-HYBRID, S13 | **15/15 gates, 13/13 mutants, COMPLETION SENTINEL: PASS** | 65.6 s, 565 MB | `454eb114…` |
| Brief II suite, after the screen edits, against **41ad509** | same files as run 3 except `utils/step1_screen.py` and the screen test | **18/18 gates, 12/12 mutants, PASS** | 84.8 s, 218 MB | `064d80c9…` |
| Brief II suite, against HEAD = snapshot `b78a15c` | — | 18/18, 12/12, but its identity gates compared against a commit that already holds the change; **not evidence** | — | `brief2_suite_after_screen.log` |

All suites ran on CPU (`CUDA_VISIBLE_DEVICES=""`), on cores 4,5 at nice 15, one at a time. Each is single-process except the mutant children, which run one at a time.

Final hashes:
- `utils/step1_screen.py` `e8bb31ac…`
- `tests/test_step1_screen.py` `aaaf3ee9…`
- `utils/window_anchor_domain.py` `657a2d85…`
- `window_optimizer.py` `ac895195…`
- `window_optimizer_bayesian.py` `e45cb44c…`
- `window_optimizer_integration_final.py` `37aa2448…`
- `tests/test_window_anchor_brief2.py` `99b5f85a…`

**Screen-only diff on top of Brief II:** `docs/drafts/anchor_v1/screen_code_on_brief2.diff` (`c512e427…`, 7 files). The Brief II tree was rebuilt in the scratchpad from `41ad509` + `code.diff`, plus the Brief II module recovered by reverting only the hook hunks; that gives back exactly `a9fc243d…`, the hash in the Brief II review request. Brief II's own `code.diff` and review-request hashes are therefore unchanged, and the screen is reviewable apart from it.

The Brief II test file changed by one hunk: the identity base moved from HEAD to `merge-base HEAD main`.

## 9. Draft manifest and registry entries (hand-landed: owner)

File `docs/drafts/anchor_v1/screen_manifest_registry_DRAFT.diff`. It is a `diff -ru` against the worktree's copies, so the manifest base includes the Brief II entries, which are also unlanded. Nothing is applied.

- `agent_manifests/window_optimizer.json`:
  - `args_map` gains `"step1-objective": "step1_objective"` and `"screen-config": "screen_config"`;
  - `default_params` gains `"step1_objective": ""` and `"screen_config": ""`;
  - `param_docs` gains both entries.
- **Why `""` and not `"bidirectional_count"`:** WATCHER emits every non-empty string as `--flag value` (`agents/watcher_agent.py:2418-2427`; empty strings are skipped, S167). With `""` a default run's argv is byte-identical to today's, and the CLI default supplies `bidirectional_count`.
- `agents/registry/parameter_registry.py`, registry `window_optimizer.py`: two `ParameterSpec`s:
  - `step1_objective`: CHOICE `bidirectional_count | screen_rank | screen_hits`, default `bidirectional_count`, `affects_runtime`;
  - `screen_config`: STRING, default `""`.

  Checked in a scratch copy: the module loads, `validate_adjustments({'step1_objective': 'screen_rank'})` gives `[]`, and `'x'` is refused.

  Adding them lets an agent **propose** a change of objective through the registry; that is an owner decision (Q3). The registry today has no entries for `derived_anchor_domain` or `anchor_holdout_guard` either, and the draft does not add them.

## 10. Unverified, open, owner questions

### 10.1 Facts found on the way

- **HEAD moved during this work.** Commit `b78a15c` "WIP snapshot 2026-11-23: anchor-brief2 (uncommitted agent work, not reviewed, not landed)", 20:44, author mmercalde. It is on this branch and holds the in-progress screen files. Both suites compared the default path against `git show HEAD:`, which after the snapshot would prove nothing. Both now compare against `merge-base HEAD main` = **41ad509** (`TFM_IDENTITY_BASE` overrides). `G-DEFAULT-OBJ` also checks that the base differs from the live file. The Brief II run before this fix (`brief2_suite_after_screen.log`) compared against the snapshot; run 2 is the valid one.
- **Variable-skip records carry `prng_type = <base>_hybrid`** (`utils/canonical_records.py:343-344`). Run H's NPZ held only one mode (12,156 rows, one `prng_type` code, one `skip_mode` code). Step 6 votes every survivor with its config `prng_type` (`prediction_generator.py:263-266`). The screen now does the same and counts both kinds (§10.2, TODO-SCREEN-4).

### 10.2 TODO-SCREEN

| ID | Item | Anchor |
|---|---|---|
| TODO-SCREEN-1 | The grader is a draft in another worktree, pinned by sha256 `c7fe48bf…` (the development pin, TASK_LOOP_ADDENDUM_3). Repin when it lands | `utils/step1_screen.py` `load_grader` |
| TODO-SCREEN-2 | The objective is bound to the Optuna study (tag) but not yet to the Step 1 search ledger's search space (Brief II added `window_anchor_domain` there; `utils/step1_search_ledger.py`) | `screen_study_check` |
| TODO-SCREEN-3 | The WF driver has no check that its grade range starts at or after a screen study's `selection_end_max`; the operator passes `--range-start` and `--previously-inspected` (§3). WF file not edited (read only) | `e2e_walk_forward.py` `build_declaration` |
| TODO-SCREEN-4 | Variable-mode survivors vote with the base `prng_type`, as Step 6 does; counted per kind in the trial record | `ScreenPlan.score` |
| TODO-SCREEN-5 | The real miner ingress (`keep_screen_records=True` on a live assembly) has not run: no rig, no GPU. Proven by AST (G-DEFAULT-WIRING) and by reading `miner/step1_ingress.py:219-220` only | `_build_test_result_from_miner` |
| TODO-SCREEN-6 | `selection_sessions` default `all` grades both sessions; the counted-session rule for confirmation (midday) is unchanged and lives in the grader declaration | `SCREEN_DEFAULTS` |
| TODO-SCREEN-7 | `vote_matrix` with `workers` > 1 uses a spawn pool; not run inside a live Step 1 process | `vote_matrix` |

### 10.3 Unverified

- Per-trial time of a multi-trial screen run on the fleet. Run H was a one-trial run, so 130.8 s may include one-off admission cost.
- Sieve time linear in seed count (§5, 2²⁶ row).
- `screen_rank` cost at run H's shape is extrapolated from the per-value cost and the cost probe (§4), not measured inside a trial.
- Whether a screen score predicts later hits at all. That is the experiment the screen makes cheap. Measuring it is the confirmation step (§7); this note claims nothing about it.

### 10.4 Owner questions (one line each)

- **Q1.** May Step 1's Optuna objective be switched, for a screen run only and with the default unchanged, to the trial's own pool hits or rank on a selection block (LL-17 Q5)?
- **Q2.** The system map (§10) forbids Step 3's `holdout_hits` as a "window"; the screen does not use that label but does choose windows by pre-holdout outcome hits: is that allowed?
- **Q3.** Should `step1_objective` go into the parameter registry, so WATCHER or an agent may propose it, or stay operator-only?

