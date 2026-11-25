# READY: walk-forward variant 02_target_hits, run H block 0 (2026-11-24). DEVELOPMENT

- **Author:** Claude (Team Alpha), window e2e, worktree `~/tfm_worktrees/e2e-map`.
- **Task:** `logs/e2e_session/TASK_WF_TARGET_HITS_BLOCK0_20261124.txt` (`fa8966f2…`).
- **Authority:** owner go 12:11, `OWNER_GO_wf_target_hits_block0_20261124.txt` (`15f3c9d8…`), "Yes train on hits": one origin, block 0, this variant only.
- **Every number is DEVELOPMENT evidence on one block,** with no verdict on TFM.

## Plain summary (written after the run)

1. **The run finished without a refusal or failure.**
   - Exit 0, wall **1:12:44** (12:32:56 to 13:45:40), 30 table rows.
   - Leak checks **L1–L6 all pass**. L3 confirms that every family's Step 5 read the shim's derived `holdout_hits` file, and that the file was built from this origin's own Step 3 output.
   - Step 3's labels are the ones described in §1: with timestamps stripped, they are byte-identical to the base run's.
2. **The pre-launch prediction held on the training side.** Cross-validation R² on the `holdout_hits` target: **catboost 0.0016**, neural_net −0.0005, xgboost −0.0072, lightgbm −0.0081. These are training scores, not hit rates.
   - The cv rule therefore chose **CatBoost**. The base and sieve-on runs chose the neural_net, at R² about 0.018 on `holdout_quality`.
3. **On block 0, every row is within about 2 null sd of the background, in all three versions.**
   - With this target, the family arms for LightGBM and the neural net fall below the background at pool 300: pooled 139 and 138 hits against 154 expected, −1.4 and −1.5 sd. The learned arm (CatBoost) sits at the background: 152 against 154.
   - **On one block these differences are mostly noise** (§3), and the table neither shows nor rules out an effect of the target.
4. **Rows above background, kept as open items** for the multi-block run and not closed by any p-value:
   - learned arm, midday, pool 100: 33 against 28.2 (p 0.20);
   - XGBoost family arm, evening, pool 300: 87 against 77.0 (p 0.10).
5. **Cost:** the same as the other two runs (§4).

## 1. Written BEFORE launch: the target

- **Field:** `holdout_hits`, Step 3's per-seed label (`full_scoring_worker.py:272-310` `compute_holdout_hits_batch`, written at `:407`).
  - It is each seed's hit count over the label draws, divided by the number of label draws.
  - Each seed is generated from position `offset = len(train)` = 2001 on (the continuation law, `:300`), so the position for label draw j is combined index 2001 + j.
- **Built from:** the run copy's `label_block_history.json` only, i.e. dataset draws **[2001, 3808)**, 1,807 draws (copy record `label_count` 1807). **Never block 0** [3808, 4402).
- **Mechanism:** the shim (`--target-field holdout_hits`) writes a derived survivors file whose `holdout_quality` field holds each row's `holdout_hits`, and the module trains on that field (`e2e_step5_shim.py` docstring item 3). Nothing else in a row changes. `holdout_hits` lies outside `features`, so it can never also be a training column.
- **Leak checks that cover it:**
  - **L1:** the label file equals dataset draws [2001, 3808) by value; train history [0, 2001).
  - **L2:** Step 3's recorded holdout input is that file, by sha256.
  - **L3:** the shim's derived file was built from this origin's Step 3 output (its original sha256 equals the labels sha256), and Step 5 read that derived file.
  - **L5 and L6:** the vote is sealed at 3808, and the grade covers [3808, 4402) only.
  - The perturbation gate of self-test W2 proves that Step 3's labels do not change when draws at or after the origin change. The variant changes only what Step 5 reads from those labels.
- **Its spread across the 12,156 survivors,** measured on the base smoke's Step 3 output. That output has the same Step 3 inputs and code as this variant, which reruns Step 3 identically apart from timestamps:

  | Hits on the label block | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | Survivors | 1,951 | 3,608 | 3,305 | 2,013 | 832 | 323 | 90 | 25 | 7 | 1 | 1 |

  - **Mean** 1.809 hits per seed. The uniform 1/1000 background predicts 1,807/1000 = 1.807.
  - **Variance/mean** 0.984. A Poisson count has 1.0; the standard error at N = 12,156 is about 0.013, so 0.984 is about −1.2 SE.
  - **11 distinct values;** the histogram matches Poisson(1.807) within a few percent per bin.
  - **Rank correlation with `holdout_quality`** (today's target): −0.0017. The learning agent measured −0.002, and mean 1.809 and variance/mean 0.984, and these figures agree.
  - **Step 5's signal gate** (`compute_signal_quality`) on this target: `weak`, confidence 0.40, `prediction_allowed: True`. So Step 6 will vote, not abstain.
- **What this implies, as a prediction, not a verdict:**
  - Counts like these are what you get if every survivor hits the drawn number at the background rate, independently. Variance/mean at or below 1 means the label block shows **no excess spread between survivors** beyond Poisson noise.
  - The variance between survivors' true rates is estimated at var − mean ≈ −0.03, so not distinguishable from zero.
  - A regressor trained on this target can only learn whatever part of that noise the features happen to predict.
  - **Prediction:** cross-validation R² near 0 for every family. The learned ranking is then close to arbitrary with respect to block 0, and block-0 hits sit near the background, like the base run.
  - **What would contradict the prediction:** learned rows clearly above background that persist over many blocks. One block cannot show that; any above-background row here stays an open item for the multi-block run.

## 2. Launch

- **Pre-launch:**
  - the 01_sieve_on run had exited (PID 39212 gone);
  - `nvidia-smi` showed compute process `sunshine` only;
  - code equal to `logs/e2e_wf/run5_code.sha256`;
  - the same pin snapshots: grader `c7fe48bf…`, vote harness `bbb04e81…`.
- **Exactly what differs from the base command** (token diff of the `.cmd` files):
  1. added `--target-field holdout_hits`;
  2. `--out-dir data/e2e_runH_wf_02_target_hits_block0` (and the matching time path).
  - Everything else is identical to the base: `taskset -c 0-5 nice -n 10`, `--gpu-allowed-procs sunshine,Xorg`, block 0 only, 5 workers. The command is in `logs/e2e_runH_wf_02_target_hits_block0.cmd`.
- **Training columns are unchanged:** 89 features, schema hash `7733d30a913545ca`, in all four frozen sidecars. Only the target differs.
- **Each sidecar's `signal_quality` describes this target,** though it still carries the name `holdout_quality` (TODO-E2E-25): target mean 0.001001 and 11 distinct values. The shim summary records `field: holdout_hits`, original survivors `a6099d7f…` and derived file `51306fa8…`.

## 3. Three-way side by side, block 0 (DEVELOPMENT; one block; rows not independent)

Cells read base / sieve on / hits target. Expected is the uniform background. Mean rank is the drawn number's rank in that pool's vote (lower is better; 500.5 is chance). The last two columns belong to the hits-target run.
- **A1** = the learned arm: the neural_net in base and sieve on, **CatBoost** in hits target.
- **A0** = unlearned, identical in all three runs, as it should be.
- **F** = one family alone.

| Session | Pool | Arm | Hits | Expected | Mean rank | Null sd | p (hits target, unadj.) |
|---|---|---|---|---|---|---|---|
| midday | 20 | A1 | 4 / 3 / 5 | 5.9 / 5.9 / 5.9 | 503.7 / 505.4 / 501.9 | 2.40 | 0.70 |
| midday | 20 | A0 | 5 / 5 / 5 | 5.9 / 5.9 / 5.9 | 502.0 / 502.0 / 502.0 | 2.40 | 0.70 |
| midday | 100 | A1 | 27 / 26 / 33 | 28.3 / 28.3 / 28.2 | 503.7 / 504.3 / 493.0 | 5.05 | 0.20 |
| midday | 100 | A0 | 28 / 28 / 28 | 28.2 / 28.2 / 28.2 | 501.3 / 501.3 / 501.3 | 5.06 | 0.55 |
| midday | 300 | A1 | 87 / 71 / 75 | 76.9 / 77.2 / 76.9 | 485.8 / 510.1 / 502.0 | 7.55 | 0.62 |
| midday | 300 | A0 | 70 / 70 / 70 | 77.0 / 77.0 / 77.0 | 512.5 / 512.5 / 512.5 | 7.55 | 0.84 |
| midday | 300 | F catboost | 68 / 69 / 75 | 77.0 / 77.1 / 76.9 | 513.9 / 514.3 / 502.0 | 7.55 | 0.62 |
| midday | 300 | F lightgbm | 87 / 83 / 61 | 76.9 / 76.9 / 76.9 | 485.1 / 490.9 / 526.8 | 7.55 | 0.99 |
| midday | 300 | F neural_net | 87 / 71 / 75 | 76.9 / 77.2 / 77.2 | 485.8 / 510.1 / 506.0 | 7.56 | 0.64 |
| midday | 300 | F xgboost | 90 / 83 / 69 | 77.0 / 77.0 / 76.9 | 480.4 / 490.6 / 514.6 | 7.55 | 0.87 |
| evening | 20 | A1 | 9 / 6 / 4 | 5.9 / 5.9 / 5.9 | 495.3 / 500.3 / 503.7 | 2.40 | 0.84 |
| evening | 20 | A0 | 4 / 4 / 4 | 5.9 / 5.9 / 5.9 | 503.7 / 503.7 / 503.7 | 2.40 | 0.84 |
| evening | 100 | A1 | 32 / 25 / 25 | 28.2 / 28.3 / 28.3 | 493.6 / 505.6 / 505.9 | 5.06 | 0.77 |
| evening | 100 | A0 | 26 / 26 / 26 | 28.2 / 28.2 / 28.2 | 504.4 / 504.4 / 504.4 | 5.05 | 0.70 |
| evening | 300 | A1 | 77 / 75 / 77 | 76.9 / 76.9 / 77.2 | 496.8 / 502.6 / 502.9 | 7.56 | 0.53 |
| evening | 300 | A0 | 80 / 80 / 80 | 76.9 / 76.9 / 76.9 | 496.2 / 496.2 / 496.2 | 7.55 | 0.36 |
| evening | 300 | F catboost | 82 / 90 / 77 | 77.0 / 77.0 / 77.2 | 493.2 / 478.2 / 502.9 | 7.56 | 0.53 |
| evening | 300 | F lightgbm | 80 / 77 / 78 | 77.0 / 76.9 / 77.0 | 497.2 / 501.9 / 497.5 | 7.55 | 0.47 |
| evening | 300 | F neural_net | 77 / 75 / 63 | 76.9 / 76.9 / 77.0 | 496.8 / 502.6 / 523.0 | 7.55 | 0.98 |
| evening | 300 | F xgboost | 76 / 71 / 87 | 77.0 / 77.0 / 77.0 | 500.9 / 510.2 / 482.3 | 7.55 | 0.10 |
| pooled | 20 | A1 | 13 / 9 / 9 | 11.8 / 11.8 / 11.8 | 499.5 / 502.8 / 502.8 | 3.40 | 0.83 |
| pooled | 20 | A0 | 9 / 9 / 9 | 11.8 / 11.8 / 11.8 | 502.8 / 502.8 / 502.8 | 3.40 | 0.83 |
| pooled | 100 | A1 | 59 / 51 / 58 | 56.5 / 56.6 / 56.5 | 498.6 / 504.9 / 499.4 | 7.15 | 0.44 |
| pooled | 100 | A0 | 54 / 54 / 54 | 56.5 / 56.5 / 56.5 | 502.9 / 502.9 / 502.9 | 7.15 | 0.66 |
| pooled | 300 | A1 | 164 / 146 / 152 | 153.8 / 154.1 / 154.1 | 491.3 / 506.3 / 502.4 | 10.68 | 0.60 |
| pooled | 300 | A0 | 150 / 150 / 150 | 153.9 / 153.9 / 153.9 | 504.3 / 504.3 / 504.3 | 10.68 | 0.66 |
| pooled | 300 | F catboost | 150 / 159 / 152 | 154.0 / 154.1 / 154.1 | 503.6 / 496.3 / 502.4 | 10.68 | 0.60 |
| pooled | 300 | F lightgbm | 167 / 160 / 139 | 153.9 / 153.8 / 153.9 | 491.1 / 496.4 / 512.1 | 10.68 | 0.93 |
| pooled | 300 | F neural_net | 164 / 146 / 138 | 153.8 / 154.1 / 154.2 | 491.3 / 506.3 / 514.5 | 10.68 | 0.94 |
| pooled | 300 | F xgboost | 166 / 154 / 156 | 154.0 / 154.0 / 153.8 | 490.6 / 500.4 / 498.4 | 10.68 | 0.43 |

**Which family the cv rule chose:**

| Run | Choice | CV ranking |
|---|---|---|
| base | **neural_net** | 0.0187; catboost 0.0178, xgboost 0.0125, lightgbm 0.0123 |
| sieve on | **neural_net** | 0.0176; catboost 0.0174, xgboost 0.0124, lightgbm 0.0120 |
| hits target | **catboost** | 0.0016; neural_net −0.0005, xgboost −0.0072, lightgbm −0.0081 |

**How to read this, and its limits:**
- **Null sd per row** at 297 draws is 2.4, 5.1 and 7.6 hits (pools 20, 100, 300); pooled it is 3.4, 7.2 and 10.7.
  - The largest deviation from background anywhere in the three tables is about 2.1 sd: F lightgbm, midday, pool 300, hits target, 61 against 76.9. In 90 rows that are not independent, deviations of this size are expected by chance alone.
  - **The differences between the three versions on one block are mostly noise.** They cannot say which target or feature set gives more hits.
- **What one block can show:** that each version runs end to end with the leak checks holding, and how far each row sits from the background. **What it cannot show:** a small or moderate effect, in either direction.
- **The pre-launch prediction** (CV R² near 0; learned hits near the background) is consistent with this block, on the training side and on block 0. One block is not a test of it; the multi-block run is.
- **Open items for the multi-block run,** recorded and not closed:
  - learned arm, midday, pool 100: 33 against 28.2;
  - F xgboost, evening, pool 300: 87 against 77.0;
  - from the base run, F xgboost, midday, pool 300: 90 against 77.0;
  - from the sieve-on run, F catboost, evening, pool 300: 90 against 77.0.
  - Each is one row on one block.

## 4. Measured cost

| Stage | Hits target | Sieve on | Base |
|---|---|---|---|
| Step 3 | 61.1 s | 61.7 s | 61.4 s |
| Step 5 NN / LightGBM / XGBoost / CatBoost | 1,775.7 / 166.4 / 245.0 / 391.5 s | 1,814.5 / 145.4 / 328.2 / 380.9 s | 1,800.3 / 145.0 / 306.7 / 382.2 s |
| 10 votes | 1,569.5 s | 1,563.0 s | 1,566.0 s |
| 30 grades | 143.1 s | 147.5 s | 144.4 s |
| total wall | **1:12:44** | 1:14:13 | 1:13:37 |
| card-heavy steps (Step 3, NN, XGBoost, CatBoost) | 2,473 s | 2,585 s | 2,551 s |
| peak card | 11,465 MiB (CatBoost) | 11,463 MiB | 11,463 MiB |
| largest single process | 983,560 kB | 983,808 kB | 983,620 kB |

- **Outputs:** `data/e2e_runH_wf_02_target_hits_block0/` (`TABLE.md`, `table.json`, `origin_0/per_draw_hits.jsonl`); log `logs/e2e_runH_wf_02_target_hits_block0.log`.
- **Not done:** no other block or variant, no commit, no push, no rigs. The main tree is untouched, and the base and sieve-on outputs were only read.
