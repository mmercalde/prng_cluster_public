# READY: walk-forward one-origin smoke on run H, block 0 (2026-11-24). DEVELOPMENT

- **Author:** Claude (Team Alpha), window e2e, worktree `~/tfm_worktrees/e2e-map`, HEAD `98a130c` (the lead's WIP snapshot).
- **Task:** `logs/e2e_session/TASK_WF_SMOKE_20261124.txt` (`d195c034…`).
- **Authority:** owner go 09:50, `OWNER_GO_wf_smoke_20261124.txt` (`045681a4…`). Its scope is one origin, block 0, variant `00_base`. The same go confirms that S179 `:546` is covered by the 10:40 ruling.
- **Every number below is DEVELOPMENT evidence:** one block, never a counted test, no verdict on TFM.
- **Not done:** no commit, no push, no rigs, no stride 4. The main tree is untouched (`git status` clean).

## Plain summary

1. **The smoke ran to the end without a refusal or failure.**
   - Exit 0, wall time **1:13:37** (09:53:33 to 11:07:11).
   - 1 origin, 30 table rows, 10 votes, 30 grades.
2. **Every runtime leak check passed** on block 0: L1 label block, L2 Step 3 inputs, L3 Step 5 inputs (4 families), L4 choice and freeze, L5 vote seal, L6 grade block.
   - **L7 does not apply.** It checks ranker arms, and `00_base` has none.
3. **What block 0 is:**
   - grade block [3808, 4402), 2006-08-20 to 2007-06-12, 594 draws (297 midday, 297 evening), never graded before;
   - labels [2001, 3808), 1,807 draws; train history [0, 2001).
   - The sieve window ends at 1112, so every draw used here lies after it.
4. **Model choice:** this was the run's first origin, so it fell back to the cross-validation rule.
   - CV scores: **neural_net 0.0187**, catboost 0.0178, xgboost 0.0125, lightgbm 0.0123. These are training R² values, not hit rates.
   - The neural net was frozen and used as the learned arm.
5. **What the table shows** (§2): on this one block, every arm's hit count is within about 1.7 null standard deviations of the k/1000 background, and every mean rank is within about 1.6 of the uniform 500.5.
   - The smallest unadjusted p in the table is 0.051 (XGBoost family arm, midday, pool 300: 90 hits against 77.0 expected), among 30 hit rows that are not independent.
   - **One block can show only a large effect** (§3). This table neither shows nor rules out a smaller one.
6. **Measured cost:** card-heavy steps (Step 3, NN, XGBoost, CatBoost) took 2,551 s (42.5 min); peak card 11,463 MiB (CatBoost); largest single process 984 MB RSS.
   - **Estimates** from these figures: **stride 4 (6 origins) ≈ 8.8 h**, of which about 4.3 h is card-heavy. **All 24 origins ≈ 36 h**, about 17 h card-heavy (§4).

## 1. Launch (as run)

**Pre-launch checks (09:52–09:53):**
- `nvidia-smi`: only compute process `sunshine` (PID 3219, 248 MiB); card 264 MiB used of 12,288.
- **Xorg was also present**, as a graphics process (type G, 4 MiB). It appeared after the reboot. The task's condition is "no other compute job", so I launched with **`--gpu-allowed-procs sunshine,Xorg`**.
  - This is a declared setting, not a code change.
  - It widens the 2026-11-14 "only sunshine" rule, which predates Xorg being listed. **Flagged for the lead.**
- **Pins:**
  - grader `c7fe48bf102daa42b284713f4b2c2508d7f21c9613a01f063be443c881fd9ace`;
  - vote harness `bbb04e81f0c11b5fdb0e90b97ec9e578858eae77e7890d2a199515073ae07bf8`;
  - vote config `d77c3ac8…`.
  - All checked against their worktrees and run from sha-verified snapshot copies in `data/e2e_runH_wf_00_base_smoke_pins/`, so an agent editing them mid-run could not change the smoke.
  - Driver and helper files equal `logs/e2e_wf/run5_code.sha256` (the code that passed self-test run 5).
- Memory 12.6 GB available; swap 0.

**The exact command** (also in `logs/e2e_runH_wf_00_base_smoke.cmd`), run with nohup:

```
taskset -c 0-5 nice -n 10 /usr/bin/time -v -o data/e2e_runH_wf_00_base_smoke.time /home/michael/venvs/torch/bin/python3 -u -B scripts/diagnostics/e2e_walk_forward.py --dataset /home/michael/skipsw_run_thr3030/daily3.json --dataset-sha256 513648160d356617c22a1e543ae1c9c65f4921ec21718989308b1f70c00768f6 --npz data/e2e_runH/copy/bidirectional_survivors_binary.npz --generation-digest 3f6a884d6536cbbdd6b23b1e8f8c7b85946d001ca1b1de49e0c2fd9dac0882d0 --scorer-config /home/michael/skipsw_run_thr3030/optimal_scorer_config.json --scorer-config /home/michael/skipsw_run_thr3030/optimal_window_config.json --sieve-window-end-index 1112 --label-window fixed --label-draws 1807 --pool-sizes 20,100,300 --grade-sessions midday,evening,pooled --model-choice prev_block_rank --vote-allow-gpu --step3-device cuda --previously-inspected 14454:16261 --previously-inspected 16261:18068 --step3-workers 5 --vote-workers 5 --grader-workers 5 --gpu-allowed-procs sunshine,Xorg --vote-harness /home/michael/tfm_worktrees/e2e-map/data/e2e_runH_wf_00_base_smoke_pins/vote/scripts/diagnostics/e2e_vote_harness.py --vote-harness-sha256 bbb04e81f0c11b5fdb0e90b97ec9e578858eae77e7890d2a199515073ae07bf8 --grader /home/michael/tfm_worktrees/e2e-map/data/e2e_runH_wf_00_base_smoke_pins/e2e_grader.py --grader-sha256 c7fe48bf102daa42b284713f4b2c2508d7f21c9613a01f063be443c881fd9ace --vote-config-sha256 d77c3ac80ad4c9175ef40f81d5605a313a0f5ab76b5e7136430449f10bd4be37 --origins 0 --out-dir data/e2e_runH_wf_00_base_smoke
```

**This differs from `runH_commands/00_base.sh` in three places,** all per the task or the snapshot rule:
- `nice -n 10` (the file has 15);
- `--origins 0` written out directly;
- the snapshot paths for the vote harness and grader, plus `--gpu-allowed-procs`.

**Outputs:**
- `data/e2e_runH_wf_00_base_smoke/`: run record `7aa7370f…`, `TABLE.md`, `table.json`, `origin_0/`;
- log `logs/e2e_runH_wf_00_base_smoke.log`; time record `data/e2e_runH_wf_00_base_smoke.time`;
- per-draw hit file `origin_0/per_draw_hits.jsonl` (`f9c334fc…`); Step 3 output `53cfab54…` (12,156 rows).

## 2. Block-0 table (DEVELOPMENT; one block; rows not independent; p one-sided, unadjusted)

**Hits against k/1000.** A1 is the learned arm (neural net); A0 is the unlearned sieve-match ordering; F rows are each family alone, at pool 300.

| Session | Pool | Arm | Graded | Hits | Expected | Null sd | p | Lift |
|---|---|---|---|---|---|---|---|---|
| midday | 20 | A1 (neural_net) | 297 | 4 | 5.89 | 2.40 | 0.84 | 0.68 |
| midday | 20 | A0 | 297 | 5 | 5.88 | 2.40 | 0.70 | 0.85 |
| midday | 100 | A1 | 297 | 27 | 28.27 | 5.06 | 0.63 | 0.96 |
| midday | 100 | A0 | 297 | 28 | 28.24 | 5.06 | 0.55 | 0.99 |
| midday | 300 | A1 | 297 | 87 | 76.95 | 7.55 | 0.10 | 1.13 |
| midday | 300 | A0 | 297 | 70 | 76.96 | 7.55 | 0.84 | 0.91 |
| midday | 300 | F catboost / lightgbm / neural_net / xgboost | 297 each | 68 / 87 / 87 / 90 | ≈77.0 | 7.55 | 0.90 / 0.10 / 0.10 / 0.051 | 0.88 / 1.13 / 1.13 / 1.17 |
| evening | 20 | A1 | 297 | 9 | 5.88 | 2.40 | 0.14 | 1.53 |
| evening | 20 | A0 | 297 | 4 | 5.89 | 2.40 | 0.84 | 0.68 |
| evening | 100 | A1 | 297 | 32 | 28.24 | 5.05 | 0.25 | 1.13 |
| evening | 100 | A0 | 297 | 26 | 28.21 | 5.05 | 0.70 | 0.92 |
| evening | 300 | A1 | 297 | 77 | 76.90 | 7.55 | 0.52 | 1.00 |
| evening | 300 | A0 | 297 | 80 | 76.93 | 7.55 | 0.36 | 1.04 |
| evening | 300 | F catboost / lightgbm / neural_net / xgboost | 297 each | 82 / 80 / 77 / 76 | ≈77.0 | 7.55 | 0.27 / 0.37 / 0.52 / 0.57 | 1.07 / 1.04 / 1.00 / 0.99 |
| pooled | 20 | A1 | 594 | 13 | 11.77 | 3.40 | 0.40 | 1.10 |
| pooled | 20 | A0 | 594 | 9 | 11.77 | 3.40 | 0.83 | 0.76 |
| pooled | 100 | A1 | 594 | 59 | 56.50 | 7.15 | 0.38 | 1.04 |
| pooled | 100 | A0 | 594 | 54 | 56.45 | 7.15 | 0.65 | 0.96 |
| pooled | 300 | A1 | 594 | 164 | 153.84 | 10.68 | 0.18 | 1.07 |
| pooled | 300 | A0 | 594 | 150 | 153.90 | 10.68 | 0.66 | 0.98 |
| pooled | 300 | F catboost / lightgbm / neural_net / xgboost | 594 each | 150 / 167 / 164 / 166 | ≈153.9 | 10.68 | 0.66 / 0.12 / 0.18 / 0.14 | 0.97 / 1.09 / 1.07 / 1.08 |

**Rank of the drawn number** (pool-300 vote; lower is better; uniform expectation 500.5). The grader's R24 value equals the driver's on every draw. About 70–77 % of draws fall outside the pool and share one mid-rank.

| Session | A1 (nn) | A0 | F catboost | F lightgbm | F neural_net | F xgboost |
|---|---|---|---|---|---|---|
| midday (n 297) | 485.8 (p 0.13) | 512.5 (0.83) | 513.9 (0.86) | 485.1 (0.12) | 485.8 (0.13) | 480.4 (0.062) |
| evening (n 297) | 496.8 (0.38) | 496.2 (0.36) | 493.2 (0.28) | 497.2 (0.39) | 496.8 (0.38) | 500.9 (0.51) |
| pooled (n 594) | 491.3 (0.16) | 504.3 (0.66) | 503.6 (0.64) | 491.1 (0.15) | 491.3 (0.16) | **490.6** (0.14) |

The p values are the grader's Monte Carlo values toward small ranks.

**What the loop rule would do with this block.** Under `prev_block_rank`, the next origin of a run would take the family with the lowest pooled mean rank here, **XGBoost (490.6)**. That is a statement about the rule only; the smoke has a single origin.

## 3. What this table can and cannot show

- **It can show** that the whole chain runs end to end on real run H data at full size (12,156 survivors), with every leak check holding, at the measured cost. It also gives one block of descriptive hits and ranks for the learned, unlearned and family arms.
- **It cannot show a small or moderate effect.** On 297 draws at pool 20 the background is about 5.9 hits with sd 2.4. A real 1.2× lift adds about 1.2 hits, well inside the noise.
- **It also cannot rule one out.**
  - The pool-300 rows (1.07–1.17 lift on some arms, p 0.05–0.18 unadjusted) are within chance at this sample size. They also come from 30 dependent rows: nested pools, shared survivors, pooled = midday + evening.
  - A lead would need the same direction across many blocks, then an untouched period.
- **One fact is worth carrying forward to stride 4, without reading anything into it:** at pool 300 the learned arm and three of the four families are above their background on midday, while A0 and CatBoost are below it. Stride 4 tests whether that holds across blocks.

## 4. Measured cost and the estimates (DEVELOPMENT)

**Measured per stage** (from the DONE records; taskset 0–5, nice 10, 5 workers):

| Stage | Wall | Peak card (total) | Peak RSS (process group) |
|---|---|---|---|
| run copy | 0.4 s | — | 8 MiB |
| Step 3 (GPU) | 61.4 s | 2,609 MiB | 4,350 MiB |
| Step 5 neural_net (GPU) | 1,800.3 s | 679 MiB | 1,098 MiB |
| Step 5 lightgbm (CPU) | 145.0 s | — | 692 MiB |
| Step 5 xgboost (GPU) | 306.7 s | 1,077 MiB | 760 MiB |
| Step 5 catboost (GPU) | 382.2 s | **11,463 MiB** | 804 MiB |
| 4 checks | 6.1 s | — | ≤ 378 MiB |
| 10 votes (3 pools × A1 + A0; 4 family arms at pool 300) | 1,566.0 s | 609 MiB | 3,709 MiB |
| 30 grades | 144.4 s | — | 472 MiB |
| **total** | **1:13:37** | 11,463 MiB | largest single process 983,620 kB |

**Card time:**
- Card-heavy steps (Step 3, NN, XGBoost, CatBoost): **2,551 s**.
- All steps that held the card, counted from the pre-checks (including the learned and NN-family votes, which used at most 609 MiB): 3,641 s.

**Note on LightGBM:** it took 145 s here against 17:41 in run H Step 5, on the same 12,156 rows and the same 20 trials × 20 folds. I record the figure and do not interpret it.

**Estimates from the measured figures:**
- **Model:**
  - Step 3, Step 5, checks and grades are taken as fixed per origin (2,846 s measured). Step 3 grows with the train length, a small effect at ≤ 100 s.
  - Votes cost more at later blocks, because Step 6 regenerates each voter up to t (`prediction_generator.py:865`). I fitted an affine cost in t to two measured points: this smoke's A1 pool-300 vote (208 s at t ≈ 4,100) and run H's (1,416 s per 1,807 draws at t ≈ 17,165, scaled to 594 draws). Assumption: every vote scales like that one.
- **Per origin:** 73.5 min (block 0, measured) to about 107 min (block 23).
- **Stride 4** (blocks 0, 4, 8, 12, 16, 20): about **8.8 h**, of which about 4.3 h is card-heavy.
- **All 24 origins:** about **36 h**, of which about 17 h is card-heavy.
- **Uncertainty:** the vote model rests on two points, measured with 5 and 6 workers. The rest repeats measured steps.

## 5. Open points for the lead

- **Xorg in the card check:** keep `--gpu-allowed-procs sunshine,Xorg` for later runs, or have the driver count only compute processes? The code is unchanged pending the decision.
- **Stride 4 is not started.** The go says the lead reports the measured cost and the owner is told first.
