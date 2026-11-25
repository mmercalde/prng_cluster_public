# RESULT · E2E follow-up grades on run H · 2026-11-23 · FOLLOW-UP, DESCRIPTIVE, NOT COUNTED

**No run is above the modelled background at this sample size.** That holds for all 14 of the
14 runs, with m = 1 each. All 14 exited 0 with every control passing.
- 10 of the 14 hit counts are above their expected hits and 4 are below.
- The runs are not independent (§4), so no combined p-value is given.
- The one counted test (A1, midday, 22 of 904 against 17.91 expected, p 0.1925) is already done.
  It stands as reported in `RESULT_E2E_GRADE_RUNH_20261123.md`. Nothing here changes or adds to it.
- **The lead stays OPEN** (draft register entry L-9, §6). Downgrading it is the owner's decision.

## Prior record

**Searches run:**
- Command: `/bin/grep -rlniE "e2e_runH_grade|counted grade|RESULT_E2E_GRADE"`.
- Scope: `docs/PROJECT_FILE_CATALOG.md`, `docs/TB_RULING_*`, `docs/PROPOSAL_*`, `docs/CHAPTER_*`,
  `docs/*_AUDIT.md`, `docs/TODO_CONSOLIDATED.md`, `docs/BACKLOG.md`, `docs/LEADS.md`,
  `docs/exchange/` and `logs/e2e_session/`, all in the main tree at `41ad509`. This is the same
  search as the counted-grade report, with the same hits.
- For the register ID: `/bin/grep -rhoE "^## L-[0-9]+"` and `\bL-(8|9|10)\b` over
  `~/distributed_prng_analysis/docs` and `~/tfm_worktrees/*/docs`.

**Found:**
- The counted grade, `docs/drafts/e2e_v1/RESULT_E2E_GRADE_RUNH_20261123.md` (this worktree, untracked).
- The task files `logs/e2e_session/TASK_FOLLOWUP_GRADE.txt` and `TASK_FOLLOWUP_VOTE.txt`.
- The owner go, `OWNER_GO_followups_20261123.txt` (14:40): "Yes", to "the same vote with pools of
  100 and 300" and "the unlearned pool on the earlier 2021-2023 draws (label block)".
- The register at `41ad509` has L-1 to L-7.
- L-8 is drafted, uncommitted, in `~/tfm_worktrees/pwc-offset/docs/LEADS.md:656` (RANGE-MINER
  two-way count).
- `docs/PROPOSAL_FINDINGS_20261113_RESOLUTION_v1_1.md:90` (WP-6) names "entries L-4 to L-9 as in
  v1.0". That range already overlaps the register's existing L-4 to L-7, which have other
  subjects. I did not resolve this.

**Labels:**
- These follow-ups and their authority: **ALREADY RECORDED** (owner go 14:40).
- The hit counts: **NEW**.
- The register entry: **NEW**, drafted as L-9.
- The WP-6 numbering overlap: **NEW** as an observation. Renumbering is the lead's or the owner's call.
- **CHANGES A RULING:** none.

**Not searched:** the host, ser8 archives, the private repository, project files outside the
repository.

## 1. Inputs and declarations

**Pool files:**
- `~/tfm_worktrees/e2e-vote/data/e2e_runH_vote_followup/DONE.json` was found at **16:32:25 PST**
  (`data/e2e_runH_grade_followup/wait.log`).
- Every sha256 it lists was re-computed and matched:
  - the 7 pool files;
  - the declaration `32de5f9a…664f`;
  - the verification `29357fca…37bd`.
- I also checked every run-record and vote-diagnostics sha256 in `verification.json`; all matched,
  and its `problems` is empty.
- Each pool file has 1,807 records (904 midday, 903 evening), one arm id, one config_id, raw pool
  length equal to the pool size, and 0 abstentions.

**Declarations:**
- Written and frozen at **16:33:07 PST**, before any outcome was read
  (`declared/run_record_grade_followup.json`, sha256
  `3ecc93dbfdff06bdbd8bee4147645005ca11d7ef87a93848454d30980a4fc6fb`).
- All files are read-only, listed in `declared/SHA256SUMS`, and re-verified OK after the runs.
- Generator: `declare.py`, sha256 `8e9f6a26…2e6e`.

**Per run:**
- One arm, declared alone (m = 1): min_k 1, max_k equal to the pool size, no comparisons.
- `upstream_unseen_from_index`: 16261 for A1 and 14454 for A0.
- Grade block: the file's block, with one session.
- `frequency_control_block`: [0, 14454) of the same session.
- Abstention not allowed; empty pool is an error.
- Flags: `--out-of-block-records exclude_and_count`, defaults otherwise (200 replicates,
  tolerance 4, seed 20261123, 4 workers).

**Grader:**
- `scripts/diagnostics/e2e_grader.py`, sha256 `a4d96fe7ac441a8befc93f55a638eb70e36e7d0cd70f24eaeb60d08b7a52169a`.
- Not edited, and the sha256 was re-checked after the runs.

**Dataset:** `daily3.json`, sha256 `513648160d35…68f6`.

**Blocks:**
- **F1** = the grade block, combined index [16261, 18068).
- **F2** = the label block, [14454, 16261). A0 only, because A1's target was built from these draws.

| Pool file (arm id) | sha256 | config_id | Cutoff |
|---|---|---|---|
| A1_F1_p100 | `38e189f57d80a4b96649a0d5e5358bb3d19f14b72b1e8cdc92dee452093e3909` | `3486fd0f…1c71` | 16261 on every record |
| A1_F1_p300 | `137dc3750d9983b4b0e87fc6639cbc8d3f4aee58f30aaf60b15a364dd36b25ee` | `160de1c8…65e3` | 16261 |
| A0_F1_p100 | `d957201332ef1f58f4678750b5f3540a6b4e9e712c4b1198e0718a68effb16b3` | `62cbc7c4…bffe` | t |
| A0_F1_p300 | `c2fff597a23388ea617629483222c96ec48ee0cabe0336c28f6e6de2c4f73256` | `608e7f5b…d78c` | t |
| A0_F2_p20 | `60043f58790e61fd3f06a06c6dc61c8b9b15d83239467df476dd0e8118b070d2` | `668224dd…8613` | t |
| A0_F2_p100 | `74314b13b2a881546ee7fb6b3506aa523d480f2a7b265ed60dacdf5f1e92dc0c` | `2c9ea597…e9ae` | t |
| A0_F2_p300 | `871c960093b27492bb2798226cdf7b9b1ac829c10cf46bdb471932ef4b2e207e` | `966e3755…e2` | t |

## 2. Results: all 14 runs, none above the modelled background

The values come from each run's `summary.json`. The first-count column was computed with the
grader's own `pb_upper_tail` on the run's graded k_t (`extract.py` `1b316ecf…`, output
`extract.json` `385f2953…`).
- Pool size: the raw pool length, then the distinct k_t as min / mean / max.
- "Graded" is graded / eligible. Abstained 0 and pending 0 everywhere. The other session's records
  (903 or 904) were excluded and counted.
- Interval: the descriptive two-sided 95 % interval on the lift.
- First count: the smallest hit count with one-sided p ≤ 0.05.
- Controls: uniform-random (z, then empirical rejection rate against exact size) and known-count
  (452 of 452 in every run).
- Frequency: hits of the top-k_t most frequent values from the same session in [0, 14454), against
  the same expected hits.

| Arm | Block | Session | Pool (distinct) | Graded | Hits | Expected | Null sd | One-sided p | Lift | 95 % interval | First count | Controls | Frequency hits |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | F2 | evening | 100 (86/95.23/100) | 903/903 | 70 | 85.997 | 8.821 | 0.9721 | 0.8140 | [0.6397, 1.0178] | 102 | pass (z 1.45; 0.070 / 0.0420) | 84 |
| A0 | F2 | evening | 300 (243/259.65/275) | 903/903 | 213 | 234.460 | 13.174 | 0.9534 | 0.9085 | [0.8032, 1.0205] | 257 | pass (z 0.96; 0.045 / 0.0482) | 220 |
| A1 | F1 | evening | 100 (87/95.29/100) | 903/903 | 82 | 86.046 | 8.823 | 0.6928 | 0.9530 | [0.7648, 1.1697] | 102 | pass (z 1.77; 0.065 / 0.0425) | 90 |
| A1 | F1 | evening | 300 (244/259.53/275) | 903/903 | 229 | 234.360 | 13.172 | 0.6700 | 0.9771 | [0.8689, 1.0916] | 257 | pass (z −0.68; 0.045 / 0.0474) | 245 |
| A0 | F2 | evening | 20 (18/19.79/20) | 903/903 | 18 | 17.874 | 4.186 | 0.5203 | 1.0070 | [0.5990, 1.5824] | 26 | pass (z −0.06; 0.035 / 0.0401) | 15 |
| A0 | F1 | midday | 300 (240/259.11/276) | 904/904 | 235 | 234.237 | 13.173 | 0.4896 | 1.0033 | [0.8940, 1.1186] | 257 | pass (z 1.57; 0.040 / 0.0465) | 234 |
| A0 | F1 | evening | 300 (244/259.09/276) | 903/903 | 235 | 233.956 | 13.165 | 0.4811 | 1.0045 | [0.8950, 1.1200] | 257 | pass (z −0.61; 0.045 / 0.0444) | 246 |
| A1 | F1 | midday | 100 (88/95.23/100) | 904/904 | 87 | 86.090 | 8.825 | 0.4754 | 1.0106 | [0.8170, 1.2323] | 102 | pass (z 0.44; 0.045 / 0.0429) | 87 |
| A0 | F1 | midday | 100 (89/95.17/100) | 904/904 | 87 | 86.031 | 8.823 | 0.4727 | 1.0113 | [0.8175, 1.2331] | 102 | pass (z 0.37; 0.045 / 0.0423) | 88 |
| A0 | F2 | midday | 20 (18/19.82/20) | 904/904 | 19 | 17.915 | 4.190 | 0.4297 | 1.0606 | [0.6409, 1.6464] | 26 | pass (z 0.62; 0.045 / 0.0410) | 16 |
| A0 | F2 | midday | 300 (243/259.42/272) | 904/904 | 241 | 234.518 | 13.178 | 0.3232 | 1.0276 | [0.9175, 1.1437] | 257 | pass (z 1.97; 0.070 / 0.0487) | 235 |
| A1 | F1 | midday | 300 (243/259.33/275) | 904/904 | 241 | 234.432 | 13.176 | 0.3209 | 1.0280 | [0.9178, 1.1441] | 257 | pass (z 1.55; 0.045 / 0.0480) | 237 |
| A0 | F1 | evening | 100 (86/95.16/100) | 903/903 | 93 | 85.931 | 8.818 | 0.2261 | 1.0823 | [0.8820, 1.3102] | 102 | pass (z 1.51; 0.065 / 0.0413) | 89 |
| A0 | F2 | midday | 100 (88/95.15/100) | 904/904 | 96 | 86.016 | 8.822 | 0.1416 | 1.1161 | [0.9130, 1.3466] | 102 | pass (z 0.87; 0.050 / 0.0422) | 78 |

The rows are sorted by one-sided p, largest first. Each grader headline reads "NULL: not above
chance after correction", verbatim, in `<run>.log`.

**Run facts:**
- Every run on CPU, about 300–320 % CPU (4 workers).
- Wall time 3.6–13.9 s per run; peak RSS 96–115 MB.
- Each run ran once. No other run or setting was used.

## 3. Plain arithmetic, no verdict

- Of 14 runs, **10 have hits above expected and 4 below**. None is equal.
- **Above:**
  - A1 F1 midday at 100 and 300;
  - A0 F1 midday at 100 and 300;
  - A0 F1 evening at 100 and 300;
  - A0 F2 midday at 20, 100 and 300;
  - A0 F2 evening at 20.
- **Below:**
  - A1 F1 evening at 100 and 300;
  - A0 F2 evening at 100 and 300.
- The smallest one-sided p is 0.1416 (A0, F2, midday, pool 100: 96 hits against 86.02 expected,
  6 hits short of the first count, 102).
- Every rate is **not above the modelled background at this sample size** (903 or 904 draws per
  run).

## 4. The runs are not independent, so there is no combined p-value

- **Same survivors.** Every pool comes from the same 12,156 run H survivors and the same vote rule.
- **Nested pools (measured).** Within an arm and block, the smaller raw pool is the leading prefix
  of the larger one on every record: 1,807 of 1,807 for each of these pairs:
  - A0 F2 20→100 and 100→300;
  - A1 F1 100→300;
  - A0 F1 100→300;
  - A1 F1 counted 20-vote file → 100.

  This was a pool-only check; no outcome was read. So the hits at 300 include every hit at 100,
  and the hits at 100 include every hit at 20.
- **Same draws.** The same 904 or 903 draws are graded across pool sizes and across A1 and A0 on F1.
- **Correlated controls.** The uniform controls share seed 20261123 per replicate, so they are
  correlated across runs too.

A sign count or a combined test over these 14 runs would treat dependent runs as independent.
**No combined p-value is given.**

## 5. What each background figure measures and assumes

Expected hits, null sd, one-sided p, the first count and the lift interval are all computed under
the grader's Poisson-binomial model.

**Measured on this data:**
- the per-draw distinct pool size k_t (min / mean / max in the table, per run, from `per_draw.jsonl`);
- the number of graded draws (903 or 904);
- the hit counts;
- the frequency-control counts. Their top values were measured on the same session in [0, 14454):
  6,707 midday rows and 7,747 evening rows.

**Assumed:**
- each draw is uniform on 000–999;
- draws are independent of each other and of the pool;
- so a draw hits with probability k_t/1000.

The uniform-random control tests the grader's arithmetic under that assumption. It does not test
the assumption itself. One descriptive check in the summary bears on independence: the lag-1
autocorrelation of residuals, per run in `summary.json` `stats`.

## 6. Status

- These follow-ups have RUN, and at this sample size none is above the modelled background:
  - **extend:** pools of 100 and 300 on the grade block, both arms, both sessions;
  - **recur:** A0 on the label block, pools of 20, 100 and 300.
- Background figures alone do not close the lead, and a follow-up being null does not close it
  either by itself (lead-handling §2, §4).
- **The lead stays OPEN.** It is drafted as **L-9** in this worktree's `docs/LEADS.md`, with the
  follow-ups that remain open. Downgrading it is the owner's decision.

**Limits, as for the counted grade:**
- one generation (run H);
- the Step 6 vote as built (combined index, skip 0);
- features and target as built;
- a frozen CatBoost ranking (cross-validation score +0.0019) for A1;
- a non-certifying harness route;
- a draft grader, not Beta-reviewed.

## 7. Output sha256s (`summary.json` / `per_draw.jsonl`, first 16 hex; full hashes in each out-dir's `SHA256SUMS`)

| Run | `summary.json` | `per_draw.jsonl` |
|---|---|---|
| A1_F1_p100_midday | `3c16413b951f36c4` | `511b9a9061fb30f6` |
| A1_F1_p100_evening | `b1640d0d05e5d3d4` | `74df6de63b1da48e` |
| A1_F1_p300_midday | `0da297c969352e10` | `b36edd8584dbcf67` |
| A1_F1_p300_evening | `cb16fc14500a0fb3` | `76191483c871b1dd` |
| A0_F1_p100_midday | `275ee490013de0d0` | `b5cdc5cf16e32aac` |
| A0_F1_p100_evening | `898d17ea7081bef8` | `0dd8ffd9f0a32d36` |
| A0_F1_p300_midday | `89bc28a190b36507` | `095571108c76cbbf` |
| A0_F1_p300_evening | `a5dd6f1d65864abd` | `50c50cea46f84967` |
| A0_F2_p20_midday | `b641f0c5350aabe6` | `70d0e58a55aa8667` |
| A0_F2_p20_evening | `777bf621f06d6d20` | `efacb23d49d82f9b` |
| A0_F2_p100_midday | `41575c92f04fdab2` | `202fa024bef7d8f6` |
| A0_F2_p100_evening | `4059ab8fa4e8f71b` | `7d21ab13d0cedb1f` |
| A0_F2_p300_midday | `47753b5d21c54f75` | `2bb69e46686844bf` |
| A0_F2_p300_evening | `509238024d1a151c` | `dee33e04a17dd430` |
