# READY: walk-forward driver (TASK_LOOP_WF + Addenda 1-3), DEVELOPMENT, 2026-11-23

- **Author:** Claude (Team Alpha), window e2e, worktree `~/tfm_worktrees/e2e-map`, branch `e2e-map` at `41ad509`. Uncommitted, not landed, not sent to Beta.
- **Tasks:** `logs/e2e_session/TASK_LOOP_WF.txt` (`75c0720d…`), `TASK_LOOP_COMMON.txt` (`ba096e50…`), `TASK_LOOP_ADDENDUM_1.txt` (`fc898ffc…`), `TASK_LOOP_ADDENDUM_2.txt` (`fdcb534c…`), `TASK_LOOP_ADDENDUM_3.txt` (window e2e, `940c7824…`). All in the main tree.
- **Every number from this driver is DEVELOPMENT evidence.** The grader runs only in development mode, so the driver can never produce a counted verdict.
- **Nothing ran on run H.** No step, vote or grade touched run H data. The only run H read was `--plan-only`, which reads the dataset's dates and sessions to cut the blocks (§5). No GPU was used. No commit, no push.

## Plain summary

**Null results first.**
- No walk-forward number exists for run H, because nothing was run there.
- The fixture hit counts are synthetic-data numbers. They test the plumbing and are not results.

**What was built.**
- **Driver:** `scripts/diagnostics/e2e_walk_forward.py`. It cuts the dataset into blocks of about 600 draws, from the earliest workable origin to the end of the dataset. For each origin block i, it:
  - builds a run copy whose train history is [0, T_i) and whose labels are [T_i, b_i);
  - runs Step 3, Step 5 (all families, through the shim) and the per-family checks;
  - freezes every family and chooses one model;
  - votes the learned arm, the unlearned arm, and optional family and ranker arms, sealed at b_i;
  - grades block i in DEVELOPMENT mode for midday, evening and pooled, scoring hits against k/1000 and the rank of the drawn number;
  - writes a per-draw hit file and one table over origins.
- **It closes the loop.** With `--model-choice prev_block_rank` (or `prev_block_hits`), origin i picks the family that did best on the previous origin's graded block, so graded results change the next round. The default stays `cv`, the run H winner rule.
- **Leak rule:** no draw at or after b_i reaches labels, training, model choice or the vote of origin i.
  - Seven runtime checks (L1–L7) enforce it, fail-closed.
  - A perturbation gate proves it. Replacing every draw from b_2 on leaves all 17 upstream fields of origin 2 byte-identical, and all of origin 1 identical, while the graded outcomes change on 432 of 432 rows.
- **Step 5 target setting** in the shim: `--target-field`. The default, `holdout_quality`, is today's behaviour with nothing rewritten. `holdout_hits` and `holdout_features.<name>` are also allowed.
- **Ranker Step 5 arms** (Addendum 2): XGBoost `rank:ndcg` and `rank:pairwise`, LightGBM `lambdarank` and CatBoost `YetiRank`.
  - One group per label draw, the survivors as items, relevance = that survivor's own Step 6 vote equals the drawn number.
  - The default is no ranker arms. Training is CPU-only until the lead gives a GPU go.
- **Learn W1-a merge applied** (Addendum 3, READY_LOOP_LEARN_W1A §6.1–6.2). My Step 3 self-test passes after it: 18 of 18.

**Tests: CPU, fixture only.**
- **Final run 5: 63 of 64 gates pass, 21:24 wall, peak RSS 740 MB.**
- **The one failure is W7:** a new `__pycache__` in the grader worktree. It holds `parameter_registry.py…pyc`, and it appeared at 19:12:59 in the GRADE agent's own `docs/drafts/loop_v1/switch_patches_20261123/`, 0.5 s after that agent wrote `parameter_registry.py.patched` there. Nothing on my side imports from that folder.
- I'm reporting it as a failure caused by another agent's write, on timestamp and content evidence, not proof. The gate is unchanged.
- Run 4 (the same code without the W9 gates): 60 of 60.

**Earliest workable origin on run H** (defaults, read from the code): **combined index 3,808, 2006-08-20** (§3). That gives **24 blocks of about 594 draws, to 2026-02-26.** Blocks 0–16 (2006 to 2020-06) overlap no previously graded block.

**Cost per origin on run H**, estimated from measured runs:
- about **75–92 min** with `cv` choice, or **95–123 min** with `prev_block_rank`;
- of that, about **47 min holds the card exclusively** (Step 3, NN, XGBoost, CatBoost);
- all 24 origins: about **30–38 h**, about **19 h of it on the card** (§6).

**Waiting for the lead's go.** No GPU job and no run H step will start without it.

## Prior record

**Searches run** (main tree `docs/`, `/bin/grep -rlIiE <pattern>` over `PROJECT_FILE_CATALOG.md TB_RULING_* TB_RULING_REQUEST_* PROPOSAL_* CHAPTER_* *_AUDIT.md TODO_CONSOLIDATED.md BACKLOG.md LEADS.md exchange/`):

| # | Pattern | Hits (read) |
|---|---|---|
| S1 | `walk[- ]?forward` | `TB_RULING_S176_WATCHER_KPI.md` Phase C `:692-727`; `TB_RULING_S177_KPI_GOVERNANCE.md:42`; `TB_RULING_S179_IMPLEMENTATION_AUTH.md:546`; `CHAPTER_13_AUDIT.md:90`, `:192`; `CHAPTER_7_AUDIT.md:395`; `TODO_CONSOLIDATED.md:225`; `PROJECT_FILE_CATALOG.md:543`; Step 1 search proposals (unrelated sense) |
| S2 | `e2e_walk_forward\|TASK_LOOP_WF` | none |
| S3 | `rolling origin\|expanding window\|origin block` | none |
| S4 | `target[_ -]field\|holdout_hits as (the )?target\|train(ed)? on hits` | `CHAPTER_6_AUDIT.md:50-83`, `:266-278` (S1: target switched from `holdout_hits` to `holdout_quality`); `CHAPTER_14_AUDIT.md`; `CHAPTER_13_LIVE_FEEDBACK_LOOP.md`; `CHAPTER_4_FULL_SCORING.md` |
| S5 | `lambdarank\|rank:ndcg\|rank:pairwise\|yetirank\|learning.to.rank\|ranker` | none |
| S6 | `mean rank\|rank of the drawn\|rank-based` | `CHAPTER_3_SCORER_META_OPTIMIZER.md:348` (Spearman objective of Step 2; unrelated) |
| S7 | `earliest (workable )?origin\|train(ing)?[- ]history (length\|end)\|moving (train\|origin)` | none |
| S8 | `evening` (read for counting or exclusion rules) | `TB_RULING_WINDOW_ANCHOR_PROPOSAL_V1_0.md:60` and `PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_SEPARATION_v1_1.md:97`, `:222`: "midday OR evening, never combined" for the **Step 1 sieve window** of the D.1 scientific run. No ruling found that excludes evening from grading |

**Documents that already decide or record the subject:**
- **S176 Phase C** (`:692-727`): a causally correct walk-forward. "No data from draw t or later may enter training, feature extraction, … prediction generation for draw t." The block frame satisfies this, because every input for a draw t in block i ends before b_i ≤ t.
- **S179 §8 item 10** (`:546`): "Do not launch the historical walk-forward until S172 Phase 7 scheduling permits it." This bears on the run H launch, not on the build. Owner question Q3.
- **Ch6 audit S1** (`:266-278`): `holdout_quality` replaced `holdout_hits` as the Step 5 target. The shim makes the target a setting whose default is today's, and changes no ruling.
- **The lead's frame** (TASK_LOOP_COMMON) and its revision (Addendum 1): setting decisions.
- **The window-anchor "never combined" rule** applies to the sieve window, not to grading. The pooled grade view does not touch it.

**Labels:**

| Item | Label |
|---|---|
| Driver, block frame, leak checks L1–L7, perturbation gate | **NEW** (built here); frame **ALREADY RECORDED** (S176 Phase C; TASK_LOOP_COMMON, Addendum 1) |
| Moving train end; earliest workable origin | **NEW** (measured from code, §3) |
| Shim `--target-field` | **NEW**; default = today; Ch6 S1 **ALREADY RECORDED**, not changed |
| Model choice from the previous graded block | **NEW** (setting, default `cv` = today's rule) |
| Rank (censored at depth R, and the grader's R24) | **NEW** here; the grader's R24 belongs to the GRADE agent |
| Ranker arms | **NEW** (Addendum 2) |
| Pooled sessions | **NEW** as a grade view; the sieve-window rule (S8) is unrelated |
| Historical walk-forward launch | **CHANGES A RULING?** S179 `:546`: owner Q3 |

**Not searched:** the host beyond the main tree's `docs/` and `logs/e2e_session/`; ser8 archives; the private repository; project files outside the repository.

## 1. What was built (all untracked unless marked)

| File | sha256 | What |
|---|---|---|
| `scripts/diagnostics/e2e_walk_forward.py` (1451 lines) | `8597d939bbdb03ec2055de628d0d8f1a2015984f3eb7535aeea8d27f672ee747` | the driver |
| `scripts/diagnostics/e2e_walk_forward_selftest.py` (695 lines) | `46c44f7ae43cfea5630212a5b1e67fe82d11623ed0888270713ca830e73af2a1` | gates W1–W9 and mutants |
| `scripts/diagnostics/e2e_step5_ranker.py` (311 lines) | `57e5a3eae36431825ddb2853adfdfce33ef48ca907a4b5fb97675a50b7054a62` | the ranker Step 5 arm |
| `scripts/diagnostics/e2e_step5_shim.py` | `7528880579a6dc73d899ccf832356338d19d2aa20fd8188599b17a3665b1470a` (was `1a7534d3…` at run H Step 5) | **edited:** `--target-field` (docstring item 3, `derive_target_file`) |
| `scripts/diagnostics/e2e_step3_harness.py` | `7e0638f7220fd630f01b7955a326bc43352df90f483de2b0dd6753541a0db78d` | **edited:** the LL-2 hunk (§7) |
| tracked: `full_scoring_worker.py`, `meta_prediction_optimizer_anti_overfit.py`, `prediction_generator.py`, `survivor_scorer.py` | `3f8d3d05…`, `16cd3ae5…`, `3cc11eb1…`, `efc08ae3…` | **Learn W1-a merge**, applied with `git apply` of `tracked_files.diff` (`92741da0…`). Each file equals ch14-learn's (`cmp`) |
| `models/feature_manifest.py`, `.json` (ignored path) | `a454e311…`, `b2fabc87…` | copied from ch14-learn (§6.1) |

**I made no edit of my own to `meta_prediction_optimizer_anti_overfit.py` or `train_single_trial.py`.** The only change to the first is the Learn merge (its hunks are listed in READY_LOOP_LEARN_W1A §1, LL-3). The Step 5 target lives in the shim, as the task asked.

### 1.1 Driver: frame and settings (defaults = today where today exists)

| Setting | Default | Meaning |
|---|---|---|
| `--sieve-window-end-index` | required | one past the last draw the sieve saw. Run H: **1112** (2002-12-10 midday + 1; midday-stream positions 25..36 = 2002-11-29..2002-12-10, checked against `optimal_window_config.json` window 12, offset 25) |
| `--range-start` / `--range-end` | auto / end | first origin = the earliest workable origin; end = the dataset end. Index or date |
| `--block-length` / `--blocks` / `--block-cut` | 600 / 0 / date | cuts move forward to a new date, so no date is split |
| `--label-window` / `--label-draws` | fixed / 1807 | labels [b−L, b), train [0, b−L): the train end moves with the origin. `expanding` = labels from `--label-start-index` (default the sieve end) |
| `--min-train-draws` | 2000 | below 2,000, Step 3's regime features are 0 (§3) |
| `--labels-after-sieve` | on | refuse a label window that overlaps the sieve window |
| `--origins` / `--origin-stride` | all / 1 | block numbers, 0-based |
| `--grade-sessions` | midday,evening,pooled | each on its own and pooled (Addendum 1 item 3) |
| `--pool-sizes` | 20 | each is `--k N --pool-size N` (vote READY §3c) |
| `--target-field` | holdout_quality | shim; or `holdout_hits`, `holdout_features.<f>` |
| `--families`, `--split-seeds`, `--trials`, `--k-folds` | all 4, 42, 20, 20 | as run H Step 5 |
| `--model-choice` | cv | `prev_block_hits` / `prev_block_rank`; the first origin falls back to cv |
| `--ranker-arms` | none | Addendum 2 (§2) |
| `--vote-value-ordering` | off | `full_vote` = the MISS agent's full ordering (§2.3) |
| `--previously-inspected` | none | recorded per arm by the grader (R22 development mode) |
| guards | sunshine only, ≥ 8 GB free, MemAvailable ≥ 2 GB | the card is checked before every GPU step; a 1 s watchdog stops the step's own process group (recorded PGID, never a pattern) |
| `--resume` | off | a step is skipped only if its DONE record has the same argv and run record and its outputs still verify |

**Per-origin directory** `origin_<i>/`:
- `split/`, `copy/`, `step3/`;
- `step5/s<seed>/<family>/`, with `.log`, `.check.json` and `.DONE.json` beside each;
- `frozen/s<seed>/<family>/` (read-only);
- `winner/s<seed>/WINNER.json`;
- `vote/<arm>/`, `grade/<arm>_<session>/`;
- `per_draw_hits.jsonl`, `origin_summary.json`.

**Top level:** `run_record.json` (written before any step), `PLAN.md`, `driver.log`, `table.json`, `TABLE.md`.

**Per-draw hit file** (`per_draw_hits.jsonl`): one row per graded draw, per arm, per session view. Its fields come from the grader's detail rows:
- draw index, date, session, drawn number, k, k_raw, hit, `vote_rank`, `vote_count`;
- the grader's R24 `rank`;
- the driver's `rank_driver`, `rank_censored` and `rank_distinct_d`;
- `origin`, `role`, `family`, `split_seed`, `session_view`, and the label `DEVELOPMENT`.

**`origin_summary.json`:**
- the block, train and labels;
- `leak_checks` L1–L7;
- `labels.survivors_with_scores_sha256`, the label-block sha256 and the train-history sha256;
- `step5` (winner, rule used, evidence blocks, cv ranking), the frozen families and the rankers, each with its sidecar and checkpoint sha256;
- `votes` (config_id, pools sha256), `grades` (hits, expected, null sd, p, lift, the driver's rank and the grader's `rank_summary`).

### 1.2 Leak rule, enforced at run time (exit 4)

| Check | What |
|---|---|
| L1 | the run copy's labels equal dataset draws [T, b_i) and its train history equals [0, T), by length and value |
| L2 | Step 3's recorded inputs are those two files (sha256) |
| L3 | Step 5's `--survivors` is this origin's Step 3 output, or the shim's derived target file made from it; `--lottery-data` is the train history |
| L4 | the choice uses Step 5 check reports and, for `prev_block_*`, only grades of blocks ending at or before b_i. The winner and every family are frozen read-only before the first vote |
| L5 | every vote is `--seal-from b_i`, with history [0, b_i) |
| L6 | the grade block is block i, `upstream_unseen_from_index = b_i`, frequency control block [0, b_i) |
| L7 | each ranker trained on this origin's label file (sha256), with T and L as declared, from this origin's Step 3 output |

A third layer comes from the vote harness itself: it refused a vote whose history no longer matched the sealed dataset (run 2, W2m `label_includes_origin`).

## 2. Ranker arms (Addendum 2): design

**Construction**, exactly as `e2e_step5_ranker.py` implements it (docstring):
- **Label draws:** y_g for g = 0..L−1 are the run copy's label file, at combined index T+g. Nothing at or after T+L is read.
- **Items:** every Step 3 survivor row r (N rows).
- **Vote:** v[r,g] = seq_r[T+g], where seq_r = `SurvivorScorer(prng_type, mod)._generate_sequence(seed_r, T+L, skip=PredictionGenerator._get_survivor_skip(row_r, default_skip))`.
  - These are the calls Step 6 makes (`prediction_generator.py:814`, `:865-868`; the scorer as built at `:267-270`).
  - Each sequence is generated once (the MISS agent's matrix method, `e2e_miss_report.py:443-476`).
- **Relevance:** rel[r,g] = 1 if v[r,g] equals y_g, else 0.
- **Features:** each row carries the Step 5 sidecar's features, in its order, loaded by Step 5's own `load_survivors_with_features`.
  - Features are per survivor, so a survivor has the same X in every group; only its relevance changes.
  - **Consequence, stated plainly:** the ranker learns which survivors' features go with voting the drawn number. This is close to what `holdout_hits` measures at the same positions, but with a ranking loss.
- **Size:**
  - All groups and all items: N·L rows. **Run H: 12,156 × 1,807 = 21,965,892 rows, × 89 float32 = 7.8 GB.**
  - The bounds are settings: `--ranker-groups G` (label draws sampled, seeded) and `--ranker-negatives M` (all positives kept, M negatives sampled per group, seeded). Rows = Σ_g (pos_g + min(M, N − pos_g)).
  - Groups with no positive are dropped (`--drop-empty-groups on`).
  - `--ranker-max-rows` (default 5,000,000) refuses anything larger.
  - Run H at M = 200: about 1,807 × (≈12 positives + 200) ≈ 383k rows ≈ 136 MB **[estimate: positives ≈ N/1000 per group]**.
- **Score:** NDCG@20 on 20 % of the groups held out (seeded). It is a training score, not a hit rate.
- **Output:**
  - The checkpoint is written in the format Step 6's own loaders read (`model_factory.py:30-32`; the XGBoost and LightGBM wrappers load a Booster; CatBoost loads into `CatBoostRegressor`). So the vote harness votes with it unchanged; the W8 gates show this for all four.
  - The sidecar is the origin's Step 5 sidecar with `model_type` and the checkpoint replaced, `optuna` removed, and a `ranking_objective` block added. Its `signal_quality` is the template's, and the sidecar says so.
- **In the driver:** each ranker is its own learned arm, `R<objective>s<seed>_o<i>_p<k>`. It is frozen, voted at every pool size and graded. It does not compete in the family choice, because NDCG and R² are not comparable.
- **Device:** `--ranker-device cpu` (default); `cuda` only with the lead's go.

## 2.3 Rank (Addendum 1 item 2)

- **Driver rule** (`RANK_RULE`):
  - The rank is taken at depth R, the largest pool. The pool's distinct values in vote order get ranks 1..d; an unvoted drawn number gets the mid-rank (d+1+1000)/2.
  - The uniform expectation is 500.5 exactly, and each draw's variance is exact.
  - **Censoring limits it.** At depth R only hits are ordered; every miss sits at the same mid-rank. So at R = 20–300 the rank adds little beyond hits. Measured on the fixture with full ordering on: grader mean rank 482.3 against the driver's censored 482.8 (run3 smoke).
- **Grader R24** (`c7fe48bf`, the lead's development pin): `--rank-summary on` gives the mean rank, the exact null, a z score and a Monte Carlo p toward small ranks.
  - With the ordering off, the grader's pool-derived rank must equal the driver's on every draw. The driver stops if they differ (W9, and mutant `rank_off_by_one`).
- **Full depth:**
  - By Step 6's own call, a full ordering costs O(R × t) per draw (`prediction_generator.py:841-846`): about 40 × the pool-300 run, roughly 16 h per 600-draw block. That is not workable.
  - The MISS agent's matrix method builds 12,156 × 3,614 in 54.7 s (Addendum 3). Its `--emit-value-ordering full_vote` (TODO-E2E-V12) is wired in as `--vote-value-ordering full_vote`, and W9 tests the plumbing on its current file.
  - **It is not pinned for run H** until the MISS agent reports it ready (Addendum 3).
- **Rank as a model-choice setting:** `--model-choice prev_block_rank`.
- **Rank as a Step 5 target:** there is no per-seed rank label. The rank of the drawn number is a property of the whole vote, not of one seed. The per-seed route to a rank objective is the ranker arms. Question Q4.

## 3. Earliest workable origin (read from the code at 41ad509 + Learn merge)

**Step 3 inputs that depend on the train history:**
1. **Per-seed features:** `scorer.extract_ml_features_batch(seeds, lottery_history=train_history)` (`full_scoring_worker.py:376-378` before the merge). Each seed is generated from position 0 over all n = len(train) draws (`survivor_scorer.py`: `skip=0`, n = len(history)).
   - Any n ≥ 1 works. Temporal-stability windows are 100 draws (`DEFAULT_TEMPORAL_WINDOW`, `:76-77`); a window shorter than 50 is skipped (`:396-400`, `:670-676`).
   - Autocorrelation lags are zero-filled when n is short (`:275-284`).
2. **14 global features:** `GlobalStateTracker(train_history)` (`full_scoring_worker.py:348`).
   - Below 100 draws, all 14 take default values (`models/global_state_tracker.py:156`).
   - Below 2,000 draws, regime detection returns 0 (`:260-265`, window 1000 × 2).
   - Temporal stability uses min(100, n//4) (`:326-330`).
3. **`holdout_hits` offset** = len(train) (`full_scoring_worker.py:300`, `:367`): the continuation law.
   - `holdout_quality` is computed on the label draws from each seed's position 0, not at that offset. This is already recorded (`CHAPTER_6_AUDIT.md:78`).

**Earliest workable origin on run H:**

| Setting | Earliest b | Date |
|---|---|---|
| defaults: train ≥ 2,000 (all 14 globals live), L = 1,807, labels after the sieve window | 3,807 → **3,808** after date snapping | **2006-08-20** |
| `--min-train-draws 1112` (train just covers the sieve window; regime features 0) | 2,919 → 2,920 after date snapping | 2005-06-02 |
| `--labels-after-sieve off`, train ≥ 100 | 1,907 → 1,908 | 2004-01-13; labels would include the sieve window's draws |

## 4. Tests (CPU, synthetic fixture only)

- **Command:** `~/venvs/torch/bin/python3 -u -B scripts/diagnostics/e2e_walk_forward_selftest.py`.
- **Fixture:** 360 draws, 80 survivors (via `e2e_step3_selftest.make_fixture`), sieve end 20, L = 40, blocks of about 40 draws, LightGBM and XGBoost on CPU, 2 trials, 3 folds.
- **Pins:** the vote harness and the grader run from sha-verified snapshots made at test start. Both agents edited their files during my runs 1–3.
- **Multi-core:** Step 3 runs 2 spawn workers, the vote 2 workers, the grader 4, and the ranker vote matrix 2 forked workers.

**Run history** (all kept under `data/`, logs under `logs/e2e_wf/`):

| Run | Result | Note |
|---|---|---|
| 1 (old frame) | 9/9 W1 pass; W2 FAIL on `step3_survivors` | the only difference was `metadata.timestamp` on 80 of 80 rows (run time); the gate now strips it, as the Step 3 self-test does. The grader then changed mid-run, so the later runs were refused (pin guard) |
| 2 (Addendum 1 frame) | 46/50 | the 4 failures, and every W4 pass after 18:12, came from the vote harness changing mid-run (`a26e39c1` → `bbb04e81`). Those W4 passes are not credited |
| 3 | stopped by me | I edited the driver mid-run; recorded in its log |
| **4** | **60/60 pass**, 35:29 wall, peak RSS 739,556 kB | data/e2e_wf_selftest_20261123_182131 |
| **5 (final)** | **63/64**, 21:24 wall, peak RSS 740,484 kB | data/e2e_wf_selftest_20261123_190004, `selftest_summary.json` `890efdb8…`; the code files equal `logs/e2e_wf/run5_code.sha256` after the run |

### 4.1 Run 5, gate by gate

- **W1 (11 gates) PASS:**
  - 3 origins, 54 rows, DEVELOPMENT everywhere, L1–L6 ok;
  - train [0, b−40) and labels [b−40, b): train lengths 138, 174, 210;
  - range [140, 360) starts at the earliest workable origin;
  - the choice is cv at origin 1, then `prev_block_rank` at origins 2–3 from the previous block;
  - frozen read-only before the vote; pooled = midday + evening; rank re-derived.
- **W2 (4) PASS:** 17 upstream fields identical under perturbation; the outcomes of block 2 differ on 432 of 432 rows; origin 1 identical.
- **W2m (3) PASS:** each leaking mutant shows up in its field (`label_block` and `step3_survivors`, `vote_history`, `step5_lottery`).
- **W3 (9) PASS:** L1, L3, L4 (two cases), L5, L6, the grader's R22 refusal in counted mode, table completeness, and the driver/grader rank mismatch ("differ on 18 draws").
- **W4 (15) PASS**, refusals with exit 2. In this run every pin was a fresh snapshot, so each refusal is for its own reason.
- **W5 PASS:** resume skipped every step after verifying it, and the table is identical.
- **W6 (6) PASS:** default target unchanged; `holdout_features.batt_ac_decay_rate` gives the sidecar's target statistics; mutant caught; 3 refusals.
- **W8 (10) PASS:**
  - all four objectives rebuild Step 6's votes exactly, row for row, at every label draw, and put 19–20 of the 20 planted good survivors on top;
  - `relevance_shift` caught; bounded sizes exact; sha refusal works;
  - in the driver, the rankers pass L7, are frozen, voted and graded; `ranker_label_shift` is killed by L7.
- **W9 (3) PASS:** the grader's R24 rank equals the driver's on every grade with the ordering off; the selfplay window holds its invariants for every block; with `full_vote`, every pool record has 1,000 value scores and the grader ranks in them (482.3 against the censored 482.8).
- **W7 (2):** main tree git status unchanged, pinned snapshots unchanged: PASS. **`__pycache__` FAIL:** caused by another agent (above).

## 5. Run H plan (`--plan-only`, dates and sessions only)

The plan is `data/e2e_runH_wf_plan_v2/` (run record `db83938503e481267df40cc88dfb2585bac2cea689699c448f3db801c3b12a4e`, `PLAN.md` `5e0e5ea2…`). It was written with the final driver and the §5.1 settings; `_plan_v1` is superseded. Its block table:
- 24 blocks of 593–596 draws;
- block 0 is [3808, 4402), 2006-08-20 to 2007-06-12; block 23 is [17475, 18068), 2025-05-06 to 2026-02-26.

Overlap with previously graded draws:
- blocks 17–20 overlap the run H label block [14454, 16261), graded by A0 in the follow-up;
- blocks 20–23 overlap the run H grade block [16261, 18068).

Both overlaps are declared with `--previously-inspected` and recorded by the grader.

## 6. Cost per origin on run H (estimates from measured runs; nothing run H was run by this driver)

| Step | Measured basis | Per origin |
|---|---|---|
| run copy + split | fixture 0.2 s; the NPZ is 58 KB | about 1 s |
| Step 3 (GPU) | 97.6 s at train 14,454, card peak 5,820 MiB (READY_H3 §9.2) | ≤ 2 min; scales with train length |
| Step 5, 4 families (NN, XGBoost, CatBoost on the card; LightGBM on CPU) | 29:35, 17:41, 5:36, 9:20 (READY_H3 §10); card peak 11,465 MiB | about 62 min, independent of origin (same 12,156 rows, same L) |
| checks | about 2 s each | < 1 min |
| votes, 3 pools × (A1 + A0), 6 workers | per 1,807 draws at t ≈ 16–18k: A1 3:50, 9:22, 23:36; A0 5:35, 11:10, 24:58 (vote READY §3b–3c) | about 26 min at the last blocks; cost grows with t (`prediction_generator.py:865`), so early blocks cost less **[the scaling is an estimate]** |
| family arms (`prev_block_*`), 4 × pool 300 | the same basis | about 31 min at the last blocks |
| grades: (2 + 4) votes × 3 views | 3.6–6.7 s per call | about 3 min |
| ranker arm, each | vote matrix ≤ 1 min (MISS: 12,156 × 3,614 in 54.7 s); training **[UNVERIFIED]** | — |

**Totals:**
- `cv` choice: about **75 min (mean over blocks) to 92 min (last blocks)** per origin.
- `prev_block_rank`: about **95–123 min**.
- 24 origins: about **30 h (`cv`) or 38 h (`prev_block_rank`)**.
- Card held exclusively: about **47 min per origin**, about **19 h** in all.
- Memory: Step 5 peak RSS ≤ 984 MB; vote group RSS about 3.5 GB (6 workers).
- **Cheaper first passes** (all settings):
  - `--origin-stride 4`: 6 origins, about 8–12 h;
  - `--families lightgbm,xgboost,catboost`: −30 min per origin;
  - `--pool-sizes 20,100`: about −40 % of the vote time.

### 5.1 Exact run H command (waiting for the lead's go; GPU)

Re-pin the vote harness and grader at launch, after their agents report ready: the current shas go in. `--vote-value-ordering full_vote` goes in only after the MISS agent reports it ready.

```
cd /home/michael/tfm_worktrees/e2e-map && source ~/venvs/torch/bin/activate
R=/home/michael/skipsw_run_thr3030
V=$(sha256sum ~/tfm_worktrees/e2e-vote/scripts/diagnostics/e2e_vote_harness.py | cut -d' ' -f1)
G=c7fe48bf102daa42b284713f4b2c2508d7f21c9613a01f063be443c881fd9ace   # lead: development pin
nohup /usr/bin/time -v -o data/e2e_runH_wf_v1.time python3 -u -B scripts/diagnostics/e2e_walk_forward.py \
  --dataset $R/daily3.json --dataset-sha256 513648160d356617c22a1e543ae1c9c65f4921ec21718989308b1f70c00768f6 \
  --npz data/e2e_runH/copy/bidirectional_survivors_binary.npz \
  --generation-digest 3f6a884d6536cbbdd6b23b1e8f8c7b85946d001ca1b1de49e0c2fd9dac0882d0 \
  --scorer-config $R/optimal_scorer_config.json --scorer-config $R/optimal_window_config.json \
  --sieve-window-end-index 1112 --pool-sizes 20,100,300 --grade-sessions midday,evening,pooled \
  --model-choice prev_block_rank --vote-allow-gpu --step3-device cuda \
  --previously-inspected 14454:16261 --previously-inspected 16261:18068 \
  --vote-harness-sha256 $V --grader-sha256 $G \
  --vote-config-sha256 d77c3ac80ad4c9175ef40f81d5605a313a0f5ab76b5e7136430449f10bd4be37 \
  --out-dir data/e2e_runH_wf_v1 > logs/e2e_runH_wf_v1.log 2>&1 &
```

Variants (Learn §6.3–6.4, Miss F-a, Addendum 2):
- add `--step3-extra-args "--per-seed-sieve-features on"`;
- add `--step5-extra-args "--enable-diagnostics --exclude-feature-groups global"`;
- add `--ranker-arms lgbm_lambdarank,xgb_rank_ndcg,xgb_rank_pairwise,catboost_yetirank --ranker-negatives 200`;
- add `--target-field holdout_hits`.

Each variant goes in its own out-dir.

### 5.2 After the 19:25 reboot (lead NOTE_zeus_reboot_20261123.txt, `01793ca5…`) and the new limit

- **Survived, checked on disk at 19:35:**
  - run 5 finished at 19:21:27, before the reboot. Its `selftest_summary.json` is `890efdb8…`, 63/64. The note's "run 5 lost" does not match the disk;
  - the run H plan v2 (`db839385…`, 19:22:24) and this note (19:22:47);
  - all five code files equal `logs/e2e_wf/run5_code.sha256`.
- **In flight at 19:25:** nothing of mine. **Rerun:** nothing.
- **New limit** (user, after the reboot; lead rule in the note):
  - every job runs under `taskset -c <assigned cores> nice -n 15`;
  - at most 2 parallel processes, and no more than about 5 busy cores on VM101 while VM100 may be in use;
  - no VM101 GPU job while VM100 is gaming.
  - **My core assignment is not in any file I can find** in `logs/e2e_session/`, so the command below takes it as `CORES`, to be filled in by the lead.
- **The run H command under the limit:**
  - prefix `taskset -c $CORES nice -n 15`;
  - `--step3-workers 2 --vote-workers 2 --grader-workers 2`;
  - the ranker vote matrix runs with the Step 3 worker count (2).
  - LightGBM in Step 5 uses every thread it is allowed. Under `taskset` that means the assigned cores only.
- **The cost rises under the limit [estimate, not measured]:**
  - votes about 3× (6 workers down to 2);
  - LightGBM Step 5 about 3× (17:41 at 662 % CPU on 8 cores);
  - NN, XGBoost and CatBoost are mostly on the card and change little;
  - per origin about **2.5–3.5 h** with `cv` choice; all 24 origins about **60–85 h**.
  - I recommend `--origin-stride 4` first (6 origins, about 15–20 h) and a measured one-origin smoke before any estimate is relied on.

```
CORES=<assigned by the lead>
cd /home/michael/tfm_worktrees/e2e-map && source ~/venvs/torch/bin/activate
nohup taskset -c $CORES nice -n 15 /usr/bin/time -v -o data/e2e_runH_wf_v1.time python3 -u -B scripts/diagnostics/e2e_walk_forward.py \
  <the §5.1 arguments> --step3-workers 2 --vote-workers 2 --grader-workers 2 \
  --out-dir data/e2e_runH_wf_v1 > logs/e2e_runH_wf_v1.log 2>&1 &
```

- **Other owner and lead files read after the reboot**, none of which changes this task:
  - `OWNER_RULING_defaults_20261123.txt` (`d7f587a8…`): defaults flip only when landed, so mine stay as they are;
  - `OWNER_RULING_selfplay_contract_not_binding_20261123.txt` (`2b3ae138…`): the kept leak rule, "a candidate is never judged on the draws it is later tested on", is what `selfplay_window` enforces (§8);
  - `LEAD_CORRECTION_mapping_prior_record_20261123.txt` (`a5ccd95d…`): it concerns the practice agent.
- **New question Q6 (lead):** which cores are mine?

## 7. Overlap and merge list

- **`e2e_step3_harness.py` (mine):**
  - `EXPECTED_SIGNATURES` for `score_survivors` gains `per_seed_sieve_features` and `fallback_metadata_merge`;
  - two argparse flags (defaults `off` and `legacy6`);
  - the task dict carries both;
  - the `score_survivors` call passes both;
  - the run record gains `declared_parameters`.
- **`e2e_step5_shim.py` (mine):** `--target-field`, `derive_target_file`, the summary `target` block, and the self-test mutant hook `E2E_SHIM_MUTANT=target_not_passed`.
- **Tracked production files:** only the Learn merge, applied verbatim (§1). None of my own hunks are in them.

## 8. Interface for selfplay (Addendum 3: keep it stable)

- **`e2e_walk_forward.selfplay_window(frame, i)`**, interface `"e2e_walk_forward.selfplay_window v1"`. `frame` is `run_record.json["frame"]`. For origin block i ≥ 1 it returns:
  - `validate_block` [b_{i−1}, b_i);
  - `validate_labels` [T′, b_{i−1}) and `validate_train` [0, T′), with T′ the label start for b_{i−1};
  - `use_block` [b_i, b_{i+1}) and `seal` = b_i;
  - the dates, `label_window`, `label_draws` and `sieve_window_end_index`.
- It raises `ValueError` outside 1..B−1. Fields are only ever added.
- **Other stable reads:** `frame.block_table[j]` = {block, start_index, end_index, rows, first_date, last_date, sessions, train, labels, overlaps_previously_inspected}, plus `label_window(frame, b)` and `cut_blocks(...)`.
- W9 checks the selfplay window's invariants for every fixture block. Selfplay reads this file read-only.

## 9. TODO, not done, UNVERIFIED

- **TODO-E2E-25:** carry the Step 5 target in the module as a declared flag. The shim carries it now, and the sidecar's `signal_quality.target_name` still reads `holdout_quality` under a rewritten target. The shim summary and the run record say which field was used.
- **TODO-E2E-WF1:** `--schema-check fail` on Step 6 (Learn §6.4) is not passed. The vote harness imports e2e-vote's `prediction_generator.py`, which does not have the LL-1 merge.
- **TODO-E2E-WF2:** the ranker's training time on run H is **[UNVERIFIED]**. So is the vote-cost scaling with t for early blocks.
- **TODO-E2E-WF3:** `holdout_hits` as a target on a 1,807-draw label block has few distinct values (run H: 10 values on 12,156 rows). The shim refuses non-finite values but not a degenerate target; the module's signal gate decides.
- **Not built:** a per-seed rank target (Q4).
- **Gate 22:** expected to fail on the untracked `.py` files under `scripts/`. I did not widen the allowlist.

## 10. Questions (one line each, only real choices)

- **Q1 (lead):** label window default: fixed 1,807 draws before each origin (run H's size, the train end moves), or expanding from the sieve end?
- **Q2 (lead):** first run H scope: all 24 origins (about 30–38 h, about 19 h on the card) or `--origin-stride 4` first (about 8–12 h)?
- **Q3 (owner):** does the 10:40 ruling (Beta holds void) cover S179 `:546` ("do not launch the historical walk-forward until S172 Phase 7 scheduling permits it"), or does the run need your go on that point?
- **Q4 (lead):** "rank as a Step 5 target": are the ranker arms what is meant? A per-seed rank label does not exist.
- **Q5 (lead):** development default for model choice: `cv` (today) or `prev_block_rank` (the loop)?
- **Q6 (lead):** which cores are assigned to window e2e for `taskset` (§5.2)?

## 11. Lead answers 20:00 (`LEAD_ANSWERS_WF_20261123.txt`, `e605b91d…`): MEANWHILE done

- **Command files:** `docs/drafts/loop_v1/runH_commands/`. Each needs `CORES`, `W` and `STAGE` = smoke | stride4 | rest, and each variant and stage writes its own new out-dir `data/e2e_runH_wf_<variant>_<stage>`. **None was run.**

  | File | sha256 | Variant |
  |---|---|---|
  | `00_base.sh` | `6f8bfb1f…` | today's features and target, four families |
  | `01_sieve_on.sh` | `d43ef0e4…` | `--step3-extra-args "--per-seed-sieve-features on"` |
  | `02_target_hits.sh` | `9f357706…` | `--target-field holdout_hits` |
  | `03_rankers.sh` | `635f0c9a…` | four ranker arms, `--ranker-negatives 200`, CPU |

- **Settings, per the lead's answers:** fixed L = 1,807 (Q1), `prev_block_rank` with the cv ranking recorded per origin in `WINNER.json` (Q5), and the smoke on block 0, which overlaps nothing graded (Q2).
- **Checked with `--plan-only`:** all 12 variant × stage combinations, on cores 6–7 at nice 15, one at a time. Each wrote a run record with the intended origins, target, ranker arms, Step 3 arguments, choice rule and worker counts (`data/e2e_runH_wf_planchk/`).
- **Bug found and fixed by that check:** with `set -u`, the venv's `activate` fails (`LD_LIBRARY_PATH: unbound variable`). The files now use `set +u; source …; set -u`.
- **Reading guide:** `docs/drafts/loop_v1/WF_FIRST_TABLE_GUIDE_20261123.md` (`ee917e55…`).
- **Stopped.** No run, no GPU.
