# READY: miss report on run H, and LL-5 in the vote harness (window h6, 2026-11-23)

**DEVELOPMENT / DESCRIPTIVE evidence, NOT COUNTED.** Every block used here has already been graded and looked at. Nothing here is a counted test. Nothing here closes or downgrades lead L-9.

## Prior record

**Searches run.** Each pattern was searched with `/bin/grep -rlIiE <pattern>` over the main tree's `docs/`, in `PROJECT_FILE_CATALOG.md`, `TB_RULING_*`, `TB_RULING_REQUEST_*`, `PROPOSAL_*`, `CHAPTER_*`, `*_AUDIT.md`, `TODO_CONSOLIDATED.md`, `BACKLOG.md`, `LEADS.md` and `exchange/`:

| # | Pattern | Hits, read |
|---|---|---|
| S1 | `miss report\|e2e_miss\|what made .* miss` | none |
| S2 | `ranked_predictions\|emit-ranked` | `CHAPTER_13_AUDIT.md:152` (Blocker 1: no producer for `predictions/ranked_predictions.json`; reader `chapter_13_orchestrator.py:875-937`); `TODO_CONSOLIDATED.md:120` (P0-C: "no `ranked_predictions.json` writer"); Ch7, Ch14 and KPI documents (file list only) |
| S3 | `rank band\|disjoint band\|1-20, 21-100` | `CHAPTER_14_AUDIT.md:261`: multi-draw evidence, minimum cohort sizes, stability and disjoint rank bands, because one draw often leaves a hit cohort of 0 or 1 |
| S4 | `per-seed .*per-run\|per_seed.*per_run\|namespace` | `CHAPTER_14_AUDIT.md`, `CHAPTER_14_TRAINING_DIAGNOSTICS.md`, `CHAPTER_4_FULL_SCORING.md`, plus unrelated `exchange/` hits (file list) |
| S5 | `L-9` | main tree: only `PROPOSAL_FINDINGS_20261113_RESOLUTION_v1_1.md:90` (an "L-4 to L-9" ID overlap). The L-9 entry itself is a draft in `~/tfm_worktrees/e2e-grader/docs/LEADS.md:658`, not landed |

I also read `~/tfm_worktrees/ch14-learn/docs/drafts/ch14_v1/PLAN_CH14_LEARNING_LOOP.md` (sha256 `d60fb3f4…`). LL-5 is at `:135` and LL-8 at `:138`.

**Already decides or records:**
- LL-5: the plan row at `:135`.
- LL-8's design for this kind of report: disjoint bands, namespaces, bootstrap and abstention (plan `:138`).
- The Ch14 audit's evidence conditions (`:261`).
- C13 Blocker 1.
- The run H grades and L-9 (grader worktree).

**Labels:**

| Item | Label |
|---|---|
| A ranked-predictions writer (LL-5) | **ALREADY RECORDED** (plan `:135`; Ch13 audit Blocker 1). **NEW** as built here, in the harness rather than in `prediction_generator.py` |
| Disjoint bands, namespaces, bootstrap, abstention | **ALREADY RECORDED** (plan `:138`; Ch14 audit `:261`). The miss report follows them |
| Full-vote rank and its exact expectation `D_t(r)/1000` | **NEW** |
| A0's key has 7 float values for 5 rounded classes (§3.4) | **NEW** (measured); it refines rev 1 F7 of the H6 READY note |
| **CHANGES A RULING** | **none** |

**Not searched:**
- the ser8 archives, the private repository, and files outside the repository;
- the bodies of `docs/exchange/` documents;
- the other loop worktrees' current drafts (the task names only this window).

## Plain summary

1. **Null results first.** The table covers 6 cells: arm, block and session.
   - In each cell, the number of draws whose drawn number ranked in each band is **not above the modelled background at this sample size**. That holds for 1–20, 21–100, 101–300, beyond 300, and never voted.
   - The largest |z| is 2.03, in A0, label block, evening, band 21–100, and there it is **below** expected: 52 against 68.12.
   - The rank-distribution test (PIT, uniform under the background) gives KS p from 0.19 to 0.80 in all 6 cells.
2. **What made pools miss, as measured.**
   - The drawn number was voted by some survivor on **every** draw: 0 "never voted" in 5,421 cell-draws.
   - It was voted **too far down**: its median first rank is 651 to 696 of 12,156.
   - Among the top N of either arm, the survivors who voted the drawn number do not differ from the rest of the top N in a way that repeats. Per cell, 1 to 4 of 51 per-seed features have an interval excluding 0. If the intervals were independent, multiplicity alone would give about 2.55; they are not independent, so this is only a reference.
   - The rank-free contrast is all outcome voters against all survivors. It flags 0 to 2 of 51 per cell, and no feature is flagged in two of the four draw sets.
3. **The big C1 differences are selection.** The task's literal contrast flags 13 to 19 of 51 features for A1. It compares top-N voters on hits with all voters on misses. Almost all of that is what the ranking selected: the top N differ from the population on 31 to 48 of 51 features. After removing selection, 1 to 4 remain (§3.3).
4. **Concentration.**
   - A1's top 20 are 20 distinct survivors with distinct scores. Vote agreement among them is 0.00097 to 0.00102 per pair, against 0.001 if the votes were independent.
   - Hits spread over the top-N survivors as a Monte Carlo of uniform draws predicts.
   - A0's top 20 are decided mostly by tie order: **16 of 20 slots come from a 39-row tie class.** At pool 100, 57 slots come from a 248-row class. At pools 100 and 300, the order between two 0.8333 classes is set by **float rounding of the sum** (§3.4).
5. **One follow-up was RUN (recurrence).** The pattern was `residual_*` features positive in two separate draw sets (§4).
   - `residual_std` and `residual_abs_mean` split 7 positive to 8 negative over the other non-abstained cells. Their rank-free intervals include 0 in all four draw sets.
   - `residual_max_abs` is positive in 10 of 15 (dependent) cells.
   - It remains an open lead. The graded follow-up is registered below and not run.
6. **LL-5 is built in the vote harness**, default `off`. With it off, pools and diagnostics are byte-identical to harness rev 4, and config_id differs only through the harness sha. With it on, the real Chapter 13 reader accepts every line and refuses a stale `draw_id`.
7. **Tests.**

   | Suite | Gates | Mutants |
   |---|---|---|
   | LL-5 | 3 of 3 pass | 3 of 3 killed |
   | H6, re-run after the edit (run 8) | 14 of 14 pass | 16 of 16 killed |
   | Miss report | 7 of 7 pass | 5 of 5 killed |

8. **Run cost.** The real-data run took 92 s on CPU. The full vote matrix used 6 forked workers and took 54.7 s. Peak RSS was 1.2 GB; group RSS peaked at 3.5 GB; free memory never fell below 8,022 MiB. No GPU was used.

## 0. Addendum 3 (lead 18:20, `TASK_LOOP_ADDENDUM_3.txt` sha256 `940c7824…`): window h6

**READY for the WF agent to pin.** Harness `scripts/diagnostics/e2e_vote_harness.py`, sha256 `bbb04e81f0c11b5fdb0e90b97ec9e578858eae77e7890d2a199515073ae07bf8`, 1,398 lines. Both new settings default to off.

**Prior record for this addendum.**
- **Read:** the grader's R24 contract (`~/tfm_worktrees/e2e-grader/scripts/diagnostics/e2e_grader.py` at `c7fe48bf…`, header `:117-142`, input format `:174-175`, `draw_ranks` `:870-907`) and `READY_LOOP_GRADE_20261123.md` (`4b98a77e…`) Addendum 1.
- The grader does not check an ordering against its pool (TODO-E2E-G10, `:890-891`). This harness now proves that before it writes.
- The frozen counted grader `a4d96fe7` reads each pool line as JSON and ignores fields it does not use (`read_pool_records`, `:359-378` of the frozen copy). It needs no change for the new fields.
- **Labels:**
  - the full-vote ordering setting: **NEW**, requested by the lead;
  - A0 key rounding: **ALREADY RECORDED** (lead answer to Miss Q2), **NEW** as built;
  - **CHANGES A RULING: none.**

### 0.1 Item 1: full per-draw ordering of all 1,000 values (`--emit-value-ordering full_vote`, TODO-E2E-V12)

**What each pool record gains:**
- `value_scores`: the grader's R24 field, 1,000 numbers. Value v scores −(rank of the first ranked survivor whose vote equals v), over **all** survivors. Values no survivor votes tie at −(n_survivors + 1).
- `rank_basis`: a string that says so.

The vote is Step 6's own rule: seq[t] of the module's generator for every survivor, with the module's per-survivor skip and the module's argsort order. The matrix is built once per run (`full_vote_matrix`, `vh:866-886`) with forked workers. It is the method of the miss report.

**Proven on every draw, or the run stops** (`attach_value_orderings`, `vh:902-937`):
1. The ordering's first len(pool) votes equal the module's pool.
2. Its top distinct values equal the pool's distinct values in first-position order.

**Identity:**
- With the setting on, `value_ordering` enters `config_id`.
- With it off, pools and diagnostics equal harness rev 5 (archived `data/e2e_loop_miss/provenance/e2e_vote_harness_rev5.py.txt`, `a26e39c1…`), and identity differs only in the harness sha.

**Gates** (`tests/test_e2e_vote_ordering.py`, `39a6cd5c…`):
- **G16a:** off = rev 5, for the learned arm and for A0.
- **G16b:** on, learned, 51 draws including the pending one.
  - Oracle 1, the **module itself** run with `--pool-size` = `--k` = all 41 survivors: `value_scores` equals −(first position) in the module's full pool, value for value, on 51 of 51 records.
  - Oracle 2, the **grader `c7fe48bf`'s own `draw_ranks`** in `record_only` mode, imported read-only (sha checked before and after): the pool's distinct values rank exactly 1..d in pool order, and every other value ranks below d, on 51 of 51 records.
- **G16c:** the per_skip arm (skip 3), 10 records.
- **G16d:** A0, field_sum + seed_asc, 30 records.
- **G18:** fail-closed.

**Mutants killed:**
- M22: last occurrence used instead of first.
- M23: matrix ignores the per-survivor skip.

**Cost per 600-draw block,** measured on a **synthetic** run H shape (`cost_probe`). This is the same shape as run H: 12,156 survivors, block [17468, 18068), a LightGBM ranking on random features, 6 workers. No run H data was read.

| | ordering off | ordering on |
|---|---|---|
| Wall | 103.4 s | 166.7 s |
| Of which, ordering | — | **45.7 s** (matrix 45.4 s; per-draw scoring and checks 0.3 s) |
| Max RSS (main) | 628 MB | 669 MB |
| `pools.jsonl` | 0.30 MB | 3.65 MB |

Source: `logs/e2e_loop_miss/selftest_ordering_run2_cost.log`, `dfb769d0…`.

- The vote part was 97.3 s and 114.1 s in the two runs. The ordering runs after the vote, so that difference is run-to-run variation, not ordering cost.
- The matrix cost grows with the sequence length, that is with the block's last vote index. For earlier origins it should be lower. That scaling is by construction and **not measured [UNVERIFIED]**.
- On real run H votes, the miss report built the 12,156 × 3,614 matrix in 54.7 s (§1.1).
- Use: the WF agent passes `--emit-value-ordering full_vote` and pins sha `bbb04e81…`. The grader then ranks with `--rank-source record_or_pool` or `record_only`.

### 0.2 Item 2: A0 key rounding (`--ranking-key-round-decimals D`, TODO-E2E-V13)

- **Default off.** It applies to `--ranking field_sum` only. Each row's key is rounded to D decimals before ranking, so that sums of one match-rate class tie. The declared tie-break then orders them (`build_key`, `vh:470-500`).
- The run record's `ranking_info` gives `distinct_before_round` and `distinct_after_round`. When on, the setting enters `config_id`.
- **Gate G17** uses a fixture key with 1e-10 float noise inside 4 classes:
  - off ranks by the raw float sums;
  - D = 6 merges 12 float values into 4 classes, and seed_asc orders inside each class;
  - each matches an independent sort, and the two orders differ.
- **Mutant M24** (rounding dropped) is killed.
- **On run H, A0's key should go from 7 float values to 5 with D = 6 [UNVERIFIED].** That comes from the miss report's class counts (§3.4), and no run with the setting was made on run H.

### 0.3 Regression after these edits (CPU, sequential)

| Suite | Result | Time, peak RSS | Log (sha256) |
|---|---|---|---|
| ordering + rounding (with cost probe) | 6/6 gates, 3/3 mutants | 288.1 s, 773 MiB | `selftest_ordering_run2_cost.log` (`dfb769d0…`); run without the probe: `selftest_ordering_run1.log` (`6a0ded81…`) |
| H6, run 9 | 14/14 gates, 16/16 mutants | 106.5 s, 640 MiB | `selftest_h6_run9.log` (`75264a02…`) |
| LL-5, run 2 | 3/3 gates, 3/3 mutants (G15a: off still = rev 4) | 14.0 s, 647 MiB | `selftest_ll5_run2.log` (`a6de131a…`) |

- The grader file was unchanged (`c7fe48bf…`), and no `__pycache__` exists in its directory after the tests.
- No GPU was used, and no run on run H data was made.

## 1. What was built

### 1.1 `scripts/diagnostics/e2e_miss_report.py` (735 lines, sha256 `ec7c7ad1…`)

It is read-only and observe-only. It writes only under `--out-dir`.

**Inputs.** Everything comes from one declared spec, `data/e2e_loop_miss/spec_run1.json` (`248310d0…`), written before the first run:
- the Step 3 rows;
- 9 H6 pool files (A1: 20, 100, 300 on the grade block; A0: 20, 100, 300 on the grade block and on the label block);
- 19 grader `per_draw.jsonl` directories.

**Outcomes come only from the grader's per-draw files.** The dataset is never opened.

**Reconstruction, proven before any analysis:**
- **The ranking.** Each arm's ranking is rebuilt from its run record argv with the harness's `build_generator`, and scored once through `_build_prediction_pool(pool_size=0)`, which casts no vote. Its X sha256 and score sha256 must equal the run record, and its effective config is compared after `build_generator` (`e2e_miss_report.py:402-453`).
- **The full vote.** Every survivor's vote on every draw comes from the module's `SurvivorScorer._generate_sequence` with the module's skip. The result is a 12,156 × 3,614 matrix over [14454, 18068) (`:455-476`).
- **Checked against the pool files.** All **16,263** pool records in the 9 files equal the first N rebuilt votes, value for value (`:550`).
- **Checked against the grader.** All **18,071** graded rows reconcile, including outcome, hit, k and pool sha (`:589`). Outcomes agree across all files.

**Definitions:**
- **R_t** is the first ranked survivor voting the drawn number (0 = none). A pool of N hits if and only if 1 ≤ R_t ≤ N.
- **Expected values** are exact per draw, given the votes and a uniform drawn number: P(R_t ≤ r) = D_t(r)/1000, where D_t(r) is the number of distinct values among the first r votes. This accounts for repeats.
- **PIT u_t** is uniform on (0, 1) under that background.
- **Bands** are disjoint.
- **Cohorts and contrasts.** Effects are in units of each feature's sd over all survivors.
  - H: top-N voters of the drawn number, on hit draws.
  - M: all voters of the drawn number, on miss draws.
  - C1 = H − M, split into the selection reference (top-N mean − M) and the remainder (H − top-N mean).
  - C1b: top-N voters against top-N non-voters.
  - C2 (rank-free and arm-free): all outcome voters, with background value 0.
- **Uncertainty.** Bootstrap over draws, B = 2000, seeded. A cohort abstains below 20 pairs, 10 draws or 5 distinct survivors.
- **Namespaces.** A column is per-run if it has the prefix `global_` (`full_scoring_worker.py:405`, `:460`) or one value across all rows (measured). Per-run columns are context only and never enter a contrast.
- **Concentration.** Score gap and tie class at N; pairwise vote agreement; how hits spread over the top N, against 2,000 uniform-draw Monte Carlo replicates.

**Shortcuts:**
- TODO-E2E-M1: the ranking line is the module's argsort (`prediction_generator.py:836`), reached through `H.ranked_order`.
- TODO-E2E-M2: the module votes only `ranked_idx[:pool_size]` (`:841`). The full vote extends `seq[t]` to all survivors. Identity with the module is proven on the top 300.

### 1.2 LL-5 in `scripts/diagnostics/e2e_vote_harness.py` (now 1,251 lines, sha256 `a26e39c1…`)

**Flags** (TODO-E2E-V11):
- `--emit-ranked-predictions {off,on}`, default `off`.
- `--ranked-predictions-depth`, default 20, because Chapter 13 reads the top 20 (`chapter_13_orchestrator.py:690`).
- `--ranked-draw-id-rule c13_fallback`: `draw_<t+1>`, from `chapter_13_diagnostics.py:163`.

**Output:** `ranked_predictions.jsonl`, one object per target, in the dict shape the C13 reader takes (`chapter_13_orchestrator.py:901-929`). Each object has:
- `draw_id`, `model_id` (model type, checkpoint sha, scores sha) and the feature-schema sha256;
- for each of the top `depth` survivors: rank, seed, row index, prediction, score, and a **name-keyed** feature map. The names come from the sidecar schema, or for unlearned arms from the row features, sorted, as at `pg:797-801`.

**Slot-to-survivor map.** It uses the module's argsort over the RankOnce scores. Each slot's raw score must equal that survivor's score, or the run stops (`ranked_line`, `vh:777-818`).

**Off is today's behaviour:**
- no file and no record key;
- the flag is not in `config_id`, so pools and config_id are the same either way.

The harness sha is in `config_id`, so config_ids of runs made after this edit differ from rev 4's. Existing pool files are untouched. They cite rev 4 (`99791061…`), which is archived as `data/e2e_loop_miss/provenance/e2e_vote_harness_rev4.py.txt` with the same sha.

**Placement.** The plan puts LL-5 in `prediction_generator.py`. That file is not edited by rule, so LL-5 lives in the harness. No file shared with other worktrees was edited: neither `meta_prediction_optimizer_anti_overfit.py` nor `train_single_trial.py`. **There are no hunks to merge.**

## 2. Tests (CPU, synthetic fixtures; multi-core where the job allows)

| Suite | Result | Time, peak RSS | Log (sha256) |
|---|---|---|---|
| `tests/test_e2e_vote_ll5.py` (`87daf203…`) | G15a, G15b, G15c pass; M19 (positional feature list), M20 (stale draw_id), M21 (slot map shifted) killed | 8.9 s, 646 MiB | `logs/e2e_loop_miss/selftest_ll5_run1.log` (`d84b90fe…`) |
| `tests/test_e2e_vote_harness.py`, run 8, after the LL-5 edit | 14/14 gates, 16/16 mutants | 85.9 s, 639 MiB | `logs/e2e_loop_miss/selftest_h6_run8.log` (`3838bdfb…`) |
| `tests/test_e2e_miss_report.py` (`ffc7250b…`) | MG1–MG7 pass; MM1 (nested bands), MM2 (global counted as seed evidence), MM3 (N/1000 ignoring repeats), MM4 (abstention ignored), MM5 (rank off by one) killed | 2.3 s, 86 MiB | `logs/e2e_loop_miss/selftest_miss_run4.log` (`edbacfe5…`) |

**What the gates cover:**
- **G15a:** rev 5 with LL-5 off against the archived rev 4, on 50 targets, and on against off with 4 workers.
- **G15b:**
  - the real `Chapter13Orchestrator.load_predictions_from_disk` accepts every line for `draw_<t+1>` and refuses it for `draw_<t>`;
  - 600 learned and 300 unlearned entries are checked against oracles: the seed's own sequence equals the prediction, and the features equal the Step 3 row;
  - the field_sum order is checked.
- **MG3:** a planted per-seed signal comes first, and per-run columns appear only as context.
- **MG7:** with uniform outcomes, C1 = selection reference + remainder, and the remainder's interval includes 0.

**Limits of the tests:**
- The miss-report test exercises the pure core functions in the same order `main` calls them, not `main` itself.
- `main`'s plumbing is guarded on real data by its own fail-closed checks: the 16,263-record pool identity and the 18,071-row grader reconciliation.

## 3. Results on run H (run3; `data/e2e_loop_miss/run3/`)

**Files:**
- `miss_report.json` (`586ff984…`)
- `per_draw_ranks.jsonl` (`1da2875c…`; per draw: rank, distinct rank, PIT, hit@20/100/300)
- `run_record.json` (`404028cb…`)

**Every background figure below** assumes each draw is uniform on 000–999 and independent of the votes. What was measured: the votes (repeats included), the ranks and the counts.

### 3.1 Bands (a, b): observed against expected; 904 midday or 903 evening draws per cell

| Cell | 1–20 | 21–100 | 101–300 | 301–12,156 | never | PIT KS p | median R |
|---|---|---|---|---|---|---|---|
| A1 grade midday | 22 / 17.91 | 65 / 68.18 | 154 / 148.34 | 663 / 669.57 | 0 / 0.00 | 0.187 | 651 |
| A1 grade evening | 16 / 17.89 | 66 / 68.15 | 147 / 148.31 | 674 / 668.64 | 0 / 0.00 | 0.794 | 696 |
| A0 grade midday | 21 / 17.91 | 66 / 68.12 | 148 / 148.21 | 669 / 669.76 | 0 / 0.00 | 0.667 | 685.5 |
| A0 grade evening | 22 / 17.89 | 71 / 68.04 | 142 / 148.02 | 668 / 669.04 | 0 / 0.00 | 0.804 | 685 |
| A0 label midday | 19 / 17.91 | 77 / 68.10 | 145 / 148.50 | 663 / 669.48 | 0 / 0.00 | 0.679 | 690 |
| A0 label evening | 18 / 17.87 | 52 / 68.12 | 143 / 148.46 | 690 / 668.54 | 0 / 0.00 | 0.495 | 669 |

- Null sd per band: 4.19, 7.94, 11.1 and 13.2.
- Cumulative counts equal the grader's hits for every cell and pool size. For example, A1 grade midday has 22, 87 and 241 hits at pools 20, 100 and 300, against 17.91, 86.09 and 234.43 expected.
- The finer rank grid (r ≤ 1, 5, 10, …, 5000) is in the JSON. Its largest excess is A1 grade midday at r ≤ 1000: 602 against 571.67, sd 14.49. The cells and grid points are dependent.

### 3.2 Which survivors voted the drawn number, hits against misses (c)

- **H (hits).** A1 midday at pool 20 has 22 pairs from 13 distinct top-20 survivors. Ranks and seeds are listed in the JSON, under `cohorts.<N>.H_survivors`.
- **M (misses).** All outcome voters on miss draws: about 7,100 distinct survivors per cell. Their ranks have a median of about 6,060 to 6,370 and a 1st percentile of about 140 to 430.
- **Abstained**, because H had fewer than 20 pairs: A1 grade evening pool 20 (16), A0 label midday pool 20 (19), A0 label evening pool 20 (18).

**Per-seed features.** 51 of the 91 columns are per-seed. 40 are per-run: 14 `global_*` columns plus 26 others that have one value in this run.

**Features with a 95 % interval excluding 0, out of 51:**

| Contrast | Per cell |
|---|---|
| C1 (literal: H − M) | A1: 17–19; A0: 3–13 |
| Selection reference (top-N mean − M) | 31–48 |
| Remainder (H − top-N mean) ≈ C1b | 1–4 |
| C2 rank-free (grade midday / grade evening / label midday / label evening) | 2 / 0 / 1 / 1 |

- The modelled reference, if no feature differed and the intervals were independent, is 2.55. The features are correlated, so this is not a test.
- No feature is flagged in more than two of the four draw sets. Where one is flagged in two, the sign is opposite for `batt_ac_lag_06`, `batt_ac_lag_07` and `batt_cs_mean_excursion`, and the same for `residual_std` and `residual_max_abs` (§4).

### 3.3 What the C1 differences are

- C1's large values (for A1: `temporal_stability_mean` +1.44, `batt_ac_sig_lag_count` +1.25, at pool 20) are carried by the selection reference: the top N are not a random sample of survivors.
- The part that separates hitting top-N survivors from the rest of the top N is the remainder. It is at the multiplicity reference level in every cell.
- Measured fact, with its limits: **at this sample size, nothing in the 51 per-seed Step 3 features separates the top-N survivors that voted the drawn number from those that did not.** This does not measure whether a different ranking would hit more often. That needs a graded variant (§5).

### 3.4 Concentration (d)

| | A1 | A0 |
|---|---|---|
| Distinct scores | 12,156 | 7 (float64); 5 rounded match-rate classes |
| Rank-N to N+1 gap, in score sd | 0.0124 / 0.0066 / 0.0011 at N = 20 / 100 / 300 | 0 at every N |
| Tie class at N (slots inside the pool) | 1 | 39 (16) at 20; 248 (57) at 100; 140 (9) at 300 |
| Distinct vote sequences in the top N | N of N | N of N |
| Vote agreement per pair | 0.00097–0.00102 | 0.00096–0.00109 |
| Hitting survivors (MC 95 % range), pool 20 | 13 (8–16) midday; 11 (8–16) evening | 12–15 (7–16) |
| Max hits by one survivor (MC range) | 3 (2–5) | 2–4 (2–5) |

- The two A0 classes that print as 0.833333 are different float64 sums: 248 rows and 140 rows. The 248 rank first. The 1.0 class is likewise 1 row plus 3 rows.
- The rev 1 F7 statement ("5 distinct values") counts rounded classes. As floats there are 7 (`numpy` on the NPZ, float32 inputs summed in float64 by the harness).

## 4. Lead handling: the follow-up that was RUN

**Object.** `residual_std` and `residual_max_abs` have a remainder interval excluding 0, with the same sign, in two separate draw sets:
- A1, grade midday, pool 20: +0.544 and +0.386;
- A0, label midday, pool 300: +0.100 and +0.121.

**Recurrence follow-up, RUN** (`data/e2e_loop_miss/followup_residual_recurrence.txt`, `5f8dc771…`): the same features in every other cell, read from run3.

| Feature | Remainder sign over non-abstained cells | Rank-free C2 |
|---|---|---|
| `residual_std` | 7 positive, 8 negative of 15 | interval includes 0 in all 4 draw sets |
| `residual_abs_mean` | 7 positive, 8 negative of 15 | interval includes 0 in all 4 draw sets |
| `residual_max_abs` | **10 positive, 5 negative of 15** | grade midday +0.018 [−0.001, +0.036]; others include 0 |

The cells share draws, and the pools are nested, so these counts are not independent.

**Status: OPEN.**
- The recurrence check did not show a consistent sign for `residual_std` or `residual_abs_mean`.
- For `residual_max_abs` it is mixed.
- No graded follow-up has tested whether ranking on this group changes pool hits. Downgrading it is the owner's decision.

**Follow-ups still OPEN.** They need a setting or a grade outside this task, so they are registered, not run:
- **(F-a)** A declared ranking variant that adds or weights the `residual_*` group. Compare it with A1 at equal k, through an LL-4-style setting comparison, on draws after 2026-02-26 or a new generation (run I).
- **(F-b)** The same recurrence check on run I's pools, once they exist.

**Proposed register text.** This is for the lead to merge into L-9 or file as its own ID. It is not written to any `LEADS.md`.
> L-9 sub-item, OPEN: `residual_*` (Step 3 per-seed) separated top-N voters of the drawn number from the rest of the top N in 2 of 4 draw sets (A1 grade midday N = 20; A0 label midday N = 300). The recurrence check RAN (read-only, run3): mixed, `residual_max_abs` positive in 10 of 15 dependent cells. Graded follow-ups F-a and F-b are OPEN.

L-9 itself is unchanged by this report.

## 5. Not done, and unverified

- No GPU job, no training, no setting change and no grade. The grader was not run.
- The real-data runs were **run1, run2 and run3**, recorded by name under `data/e2e_loop_miss/`:
  - **run1** failed closed: its config check compared the config before `build_generator`, which sets `require_feature_schema=False` for unlearned arms. My bug, fixed at `e2e_miss_report.py:418-421`.
  - **run2** produced the same numbers as run3, without the selection split.
  - **run3** is the reported one.
- **C1's definition of M** ("all voters of the drawn number on miss draws") is my reading of the task's "survivors who voted for the drawn number on misses". The split in §3.3 shows how much of C1 depends on it.
- **The 0.01-sd window** in the concentration table is a declared spec value, not derived.
- **I broke the no-`tail` rule once**, on a 2-second test re-run. It was repeated with `tee` to `selftest_miss_run2.log`.

## 6. Owner questions (one line each, real choices only)

1. Should graded follow-up F-a (a `residual_*` ranking variant against A1 at equal k, on an untouched period) be scheduled when run I or post-2026-02-26 draws exist?
2. Should A0's unlearned key be rounded to its match-rate classes before ranking, as a declared setting? Today float rounding orders 248 against 140 rows at pools 100 and 300.
