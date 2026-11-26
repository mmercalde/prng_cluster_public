# X-STEP2-WIRING · R1 · OPINION_REQUEST · Step 2 settings to Steps 3 and 5

Sent 2026-11-22 on the owner word at 18:44. This is an opinion request: it is non-blocking, it has one round, and nothing waits on it. The owner wants to know what you think.

## Prior record

**Searches run** (main tree `41ad509`, read-only; worktree `pwc-offset` at `ab2d330`, 2026-11-22):
- `/bin/grep -lIiE "optimal_scorer_config"` over `docs/PROJECT_FILE_CATALOG.md`, `docs/TB_RULING_*`, `docs/TB_RULING_REQUEST_*`, `docs/PROPOSAL_*`, `docs/CHAPTER_*`, `docs/*_AUDIT.md`, `docs/TODO_CONSOLIDATED.md`, `docs/BACKLOG.md`, `docs/LEADS.md`, `docs/exchange/*.md`, `docs/exchange/*/*.md`. It returned 20 files; the relevant ones are listed below.
- `/bin/grep -lIiE "X-STEP2|step2.?wiring|wire.{0,30}step ?[35]"` over `docs/exchange/`, `BACKLOG`, `TODO_CONSOLIDATED`. No hit, so there is no earlier exchange on this subject.
- `/bin/grep -n "optimal_scorer_config"` over the root `*.py`, and `suggest_*` in the Step 5 files.

**Documents that decide or record the subject:**

| Ref | Document | What it records |
|---|---|---|
| R1 | `docs/SESSION_CHANGELOG_20260222_S108.md:92-122` | The design: `optimal_scorer_config.json` "is a shared config serving multiple pipeline steps … All 11 Optuna dimensions are valid". Its table puts residue mods, offset, temporal and confidence in Step 3 (`SurvivorScorer` `:100-111`) and the NN settings in Step 5 (lines 145-160). "Do NOT remove." |
| R2 | `docs/CHAPTER_3_ALIGNMENT_AUDIT.md:163-189` (F6) | Only `residue_mod_1/2/3` and `max_offset` reach the Step 2 objective (`scorer_trial_worker.py:380-391`). |
| R3 | `docs/CHAPTER_3_SCORER_META_OPTIMIZER.md` §17 B2 (`:1477`) | No Step 4 or 5 code reads the config. Step 5 samples its own values in an incompatible encoding (`generate_anti_overfit_jobs.py:29-39`). |
| R4 | `docs/CHAPTER_4_AUDIT.md:111-135` (B5) | Step 3 never passes the config: `full_scoring_worker.py:344`; `generate_step3_scoring_jobs.py:201-209`, `:241-248`. Applying it renames `residue_{mod}_*` and breaks `holdout_quality.py:61-82`. |
| R5 | `docs/S107_session_log.md:96` | `max_offset` scale flaw, "Needs TB approval before v4.5". **Parked by the owner (O2).** |
| R6 | `docs/CHAPTER_13_LIVE_FEEDBACK_LOOP_v1_1.md:878-882` | Lists `hidden_layers`, `dropout`, `learning_rate` and `batch_size` as sourced from `optimal_scorer_config.json`, with ranges that differ from Step 2's. |
| R7 | `docs/drafts/PLAN_STEP2_REPAIR_20261122.md` §2 (worktree draft) | Alpha's table of the 11 parameters, showing where each is dropped. |

**Owner rulings for this request** (verbatim files in `~/distributed_prng_analysis/logs/skipsw_session/`):
- `OWNER_RULING_step2_O1_20261122.txt` (sha `be989c63…`), **O1 = wire them.** "yes , we should fix step 2 so that the remaing steps can utilie the data". Nothing is removed.
- `OWNER_RULING_step2_O2_20261122.txt` (sha `67b189b8…`), **O2 = leave the v4.3 objective as it is.** Today's runs are minuscule parts of the seed space and are not grounds for any change. The S107 `max_offset` item stays parked.
- **Owner, 16:39 and 16:42:**
  - constant-skip survivors are never dropped;
  - no include or remove decision rests on test runs;
  - full searches will have both constant and variable survivors.

**Labels:**
- Wiring itself is **ALREADY RECORDED** (R1, R3, R4) and now **owner-ruled** (O1).
- The questions below are **NEW** as questions to you.
- Nothing here **CHANGES A RULING**.

**Not searched:** the host beyond the main tree, the ser8 archives, the private repository, and project files outside the repository.

## The subject

The owner has ruled that the eleven Step 2 settings are to reach their consumers as S108 recorded (R1). Today none of them reaches any later step (R3, R4).

We are asking your opinion on how to do that safely, before any code is written.

## Please read

**Step 2 (producer):**
- `scorer_trial_worker.py`, the search space and the `:380-391` mask
- `generate_scorer_jobs.py`

**Step 3:**
- `full_scoring_worker.py`: `:344` and the argument parser `:522-550`
- `generate_step3_scoring_jobs.py`: `:201-209` and `:241-248`
- the `survivor_scorer.py` constructor, `:97-111`
- feature naming at `survivor_scorer.py:631` and `:666-667`
- `holdout_quality.py:61-82`
- `generate_full_scoring_jobs.py:25`, which already takes a `--config` argument. We have not established whether it is current.

**Step 5:**
- `generate_anti_overfit_jobs.py:29-39`, the list-string encoding sampled per job
- `anti_overfit_trial_worker.py:144-160`, which consumes that encoding
- `meta_prediction_optimizer_anti_overfit.py:2463-2476`, a second NN sampler with different ranges: `n_layers` 2–4, `layer_i`, dropout 0.2–0.6, batch 256/512/1024

**Step 6 and Chapter 13:**
- `prediction_generator.py`, for what it reads from the Step 3 and Step 5 outputs
- `docs/CHAPTER_13_LIVE_FEEDBACK_LOOP_v1_1.md:870-890`

## Questions

1. **Safest wiring.** From your reading of Step 3 and Step 5, what is the safest way to carry the Step 2 values to them? For example, a config path in the job, explicit arguments, or a run-scoped copy of the file. Should a consumer fail closed when the config is absent or stale?
2. **What breaks downstream when Step 3 honours them?** Please cover:
   - the 91/89 feature-name contract, since residue feature names follow the modulus;
   - `holdout_quality`, which reads `residue_1000/8/125_*` by name;
   - Step 5's inputs and any saved model metadata;
   - Step 6;
   - the Chapter 13 relearn path (3→5→6) and its parameter table (R6).
3. **Step 5's relationship to Step 2's values.** Two Step 5 samplers exist (listed under Step 5 above), and Step 2 also searches these four settings. Should Step 2's values be fixed inputs, starting points such as an enqueued first trial, or bounds for Step 5's own search? What does each choice cost?
4. **Order and tests.** In what order would you land the changes? Which tests would show each step is honouring the values? For example, a value placed in the config is visible in that step's effective parameters, with a mutant that drops it going red.

## What is NOT asked

- No change to the v4.3 objective. The S107 `max_offset` item stays parked (O2).
- No removal of any parameter (O1).
- No verdict on survivors or on any population.
  - Constant-skip survivors are never dropped.
  - No include or remove decision rests on test runs.
- No state-recovery framing. TFM is judged only by pool hit rate against `k/1000` on unseen draws.

Please keep to real-world cases. Label each concern **LIVE** (it happens on the current code and data path) or **THEORETICAL** (it could happen, but there is no current path).

## Context only: today's diagnostic harness

- **What ran:** 100 Step 2 trials, run in-process on run H's NPZ (12,156 rows, sha `3f6a884d…`, all `skip_mode` 1), on CPU, in 15 s. Results: 84 scored, 16 scored −1.0, 0 raised; best 0.20 (`data/step2_diag_runH_20261122/run_record.json`).
- **Limits:** the run is non-certifying; it is not WATCHER or the fleet. It covers one small, variable-skip-only slice of the seed space, which is why enrichment is 0. Per O2 it is not grounds for changing the objective, and it says nothing about Steps 3 and 5.
