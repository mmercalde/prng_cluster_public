# READY: walk-forward variant 01_sieve_on, run H block 0 (2026-11-24). DEVELOPMENT

- **Author:** Claude (Team Alpha), window e2e, worktree `~/tfm_worktrees/e2e-map`.
- **Task:** `logs/e2e_session/TASK_WF_SIEVE_ON_BLOCK0_20261124.txt` (`fa976736…`).
- **Authority:** owner go 11:14, `OWNER_GO_wf_sieve_on_block0_20261124.txt` (`53de14bf…`): step 1 only, the same block 0.
- **Every number is DEVELOPMENT evidence on one block:** never a counted test, no verdict on TFM.
- **Not done:** no other block, no commit, no push, no rigs. The main tree is untouched. The base outputs (`data/e2e_runH_wf_00_base_smoke`) were only read.

## Plain summary

1. **The run finished without a refusal or failure.**
   - Exit 0, wall **1:14:13** (11:15:43 to 12:29:56).
   - 30 table rows; leak checks **L1–L6 all pass**. L7 does not apply, as there are no ranker arms.
2. **What changed: the models saw two more columns per survivor,** `forward_matches` and `reverse_matches` (each seed's own sieve match rates).
   - They **trained on 91 features, schema hash `92bc58a65ea76724`**, against the base's 89 and `7733d30a913545ca`.
   - This was predicted before launch from manifest v2, and confirmed afterwards from the Step 3 output (93 extracted, schema v2) and from all four frozen sidecars.
3. **Model choice:** the cross-validation rule chose the **neural_net** again.
   - CV scores: neural_net 0.0176 (base 0.0187), catboost 0.0174 (0.0178), xgboost 0.0124 (0.0125), lightgbm 0.0120 (0.0123). These are training R² values, not hit rates.
4. **The unlearned arm (A0) is identical to the base** on every row. That is expected: its ordering does not use the model's features. It is a control that only the learned part changed.
5. **Side by side** (§2): with the sieve columns, most learned and family rows have a few hits fewer than the base, and CatBoost a few more.
   - Pooled, pool 300: learned arm 146 against 164 in the base, expected ≈154. Family arms: catboost 159 against 150, lightgbm 160 against 167, xgboost 154 against 166. The neural_net family row equals the learned arm, because the neural net is the winner.
   - **On one block these differences are mostly noise.** The null sd at pool 300 pooled is 10.7 hits.
   - Both versions sit within about 1.7 null sd of the background on every row (within 1.3 on the pooled pool-300 rows). The two versions' pools come from different models that partly share survivors, so the differences are not independent of each other either.
   - The table neither shows nor rules out an effect of the sieve columns.
6. **Measured cost is the same as the base:** 1:14:13 against 1:13:37; card-heavy steps 2,585 s; card peak 11,463 MiB (CatBoost); largest process 984 MB RSS.

## 1. Launch

- **Pre-launch:** `nvidia-smi` showed compute process `sunshine` only (248 MiB; Xorg allowed as declared, per the lead's 11:16 decision).
- **The same pins as the base smoke:** the snapshot copies in `data/e2e_runH_wf_00_base_smoke_pins/`, read only.
  - grader `c7fe48bf…`; vote harness `bbb04e81…`. Its linked modules `prediction_generator.py` `53e9fe71…` and `survivor_scorer.py` `a53a873f…` are unchanged against the base run's vote record.
  - Driver and helpers unchanged (`logs/e2e_wf/run5_code.sha256` OK).
- **Exactly what differs from the base command** (token diff of `logs/e2e_runH_wf_00_base_smoke.cmd` against `logs/e2e_runH_wf_01_sieve_on_block0.cmd`):
  1. added `--step3-extra-args "--per-seed-sieve-features on"`;
  2. `--out-dir data/e2e_runH_wf_01_sieve_on_block0` (and the `/usr/bin/time` output path to match).
  - Everything else is identical: `taskset -c 0-5 nice -n 10`, block 0 only (`--origins 0`), 5 workers, the same dataset, NPZ, configs, pools, sessions, choice rule, guards and pins.
- **Outputs:**
  - `data/e2e_runH_wf_01_sieve_on_block0/`, with `TABLE.md`, `table.json` and `origin_0/per_draw_hits.jsonl`;
  - Step 3 output `3ec6b86c…` (12,156 rows; the base's was `53cfab54…`);
  - log `logs/e2e_runH_wf_01_sieve_on_block0.log`.

## 2. Side by side with the base, block 0 (DEVELOPMENT; one block; rows not independent)

The block is [3808, 4402), 2006-08-20 to 2007-06-12: 297 midday, 297 evening, 594 pooled. Expected is the uniform k/1000 background for that row. Mean rank is the drawn number's rank in that pool's vote: lower is better, 500.5 is chance. A1 is the learned arm (the neural_net in both runs), A0 the unlearned arm, F a family on its own.

| Session | Pool | Arm | Hits base | Hits sieve on | Δ | Expected base / on | Mean rank base | Mean rank on |
|---|---|---|---|---|---|---|---|---|
| midday | 20 | A1 | 4 | 3 | −1 | 5.89 / 5.89 | 503.7 | 505.4 |
| midday | 20 | A0 | 5 | 5 | 0 | 5.88 / 5.88 | 502.0 | 502.0 |
| midday | 100 | A1 | 27 | 26 | −1 | 28.27 / 28.30 | 503.7 | 504.3 |
| midday | 100 | A0 | 28 | 28 | 0 | 28.24 / 28.24 | 501.3 | 501.3 |
| midday | 300 | A1 | 87 | 71 | −16 | 76.95 / 77.18 | 485.8 | 510.1 |
| midday | 300 | A0 | 70 | 70 | 0 | 76.96 / 76.96 | 512.5 | 512.5 |
| midday | 300 | F catboost | 68 | 69 | +1 | 77.01 / 77.06 | 513.9 | 514.3 |
| midday | 300 | F lightgbm | 87 | 83 | −4 | 76.89 / 76.89 | 485.1 | 490.9 |
| midday | 300 | F neural_net | 87 | 71 | −16 | 76.95 / 77.18 | 485.8 | 510.1 |
| midday | 300 | F xgboost | 90 | 83 | −7 | 76.99 / 77.02 | 480.4 | 490.6 |
| evening | 20 | A1 | 9 | 6 | −3 | 5.88 / 5.88 | 495.3 | 500.3 |
| evening | 20 | A0 | 4 | 4 | 0 | 5.89 / 5.89 | 503.7 | 503.7 |
| evening | 100 | A1 | 32 | 25 | −7 | 28.24 / 28.31 | 493.6 | 505.6 |
| evening | 100 | A0 | 26 | 26 | 0 | 28.21 / 28.21 | 504.4 | 504.4 |
| evening | 300 | A1 | 77 | 75 | −2 | 76.90 / 76.94 | 496.8 | 502.6 |
| evening | 300 | A0 | 80 | 80 | 0 | 76.93 / 76.93 | 496.2 | 496.2 |
| evening | 300 | F catboost | 82 | 90 | +8 | 76.99 / 76.99 | 493.2 | 478.2 |
| evening | 300 | F lightgbm | 80 | 77 | −3 | 77.00 / 76.93 | 497.2 | 501.9 |
| evening | 300 | F neural_net | 77 | 75 | −2 | 76.90 / 76.94 | 496.8 | 502.6 |
| evening | 300 | F xgboost | 76 | 71 | −5 | 76.96 / 77.00 | 500.9 | 510.2 |
| pooled | 20 | A1 | 13 | 9 | −4 | 11.77 / 11.78 | 499.5 | 502.8 |
| pooled | 20 | A0 | 9 | 9 | 0 | 11.77 / 11.77 | 502.8 | 502.8 |
| pooled | 100 | A1 | 59 | 51 | −8 | 56.50 / 56.61 | 498.6 | 504.9 |
| pooled | 100 | A0 | 54 | 54 | 0 | 56.45 / 56.45 | 502.9 | 502.9 |
| pooled | 300 | A1 | 164 | 146 | −18 | 153.84 / 154.12 | 491.3 | 506.3 |
| pooled | 300 | A0 | 150 | 150 | 0 | 153.90 / 153.90 | 504.3 | 504.3 |
| pooled | 300 | F catboost | 150 | 159 | +9 | 153.99 / 154.06 | 503.6 | 496.3 |
| pooled | 300 | F lightgbm | 167 | 160 | −7 | 153.89 / 153.82 | 491.1 | 496.4 |
| pooled | 300 | F neural_net | 164 | 146 | −18 | 153.84 / 154.12 | 491.3 | 506.3 |
| pooled | 300 | F xgboost | 166 | 154 | −12 | 153.95 / 154.03 | 490.6 | 500.4 |

**cv rule's choice:** neural_net in both runs. The full variant table, with null sd, p and lift per row, is `data/e2e_runH_wf_01_sieve_on_block0/TABLE.md`.

**How to read this** (and how not to):
- **Null sd per row,** at 297 draws per session: 2.4 (pool 20), 5.1 (pool 100), 7.6 (pool 300). Pooled: 3.4, 7.2, 10.7.
  - The largest drop, learned arm midday pool 300 (87 to 71), is 2.1 null sd, within the range one block produces. The base's 87 was itself +1.3 sd above background, and the variant's 71 is −0.8 sd below.
- **Neither version is clearly away from the background** on any row (all within about 1.7 sd). So the side-by-side cannot say whether the sieve columns help or hurt hit rate.
- **What could say it:** the same comparison over many blocks (stride 4, then all), read as consistency of direction, and then an untouched period.
- **One cross-check, stated as fact:** both runs' learned models are neural nets. The F neural_net rows equal the A1 rows in each run, as they should. A0 is unchanged, as it should be.

## 3. Measured cost (same settings as the base)

| Stage | Sieve on | Base |
|---|---|---|
| Step 3 (GPU) | 61.7 s, card 2,609 MiB, group RSS 4,362 MiB | 61.4 s |
| Step 5 NN / LightGBM / XGBoost / CatBoost | 1,814.5 / 145.4 / 328.2 / 380.9 s | 1,800.3 / 145.0 / 306.7 / 382.2 s |
| 4 checks | 6.1 s | 6.1 s |
| 10 votes | 1,563.0 s | 1,566.0 s |
| 30 grades | 147.5 s | 144.4 s |
| total wall | **1:14:13** | 1:13:37 |
| card-heavy steps (Step 3, NN, XGBoost, CatBoost) | **2,585 s** | 2,551 s |
| peak card | 11,463 MiB (CatBoost) | 11,463 MiB |
| largest single process | 983,808 kB | 983,620 kB |

The stride-4 and full-run estimates in `READY_WF_SMOKE_20261124.md` §4 apply unchanged: about 8.8 h for 6 origins and about 36 h for 24, per variant.
