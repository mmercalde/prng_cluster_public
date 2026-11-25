# RESULT · E2E grade on run H · 2026-11-23 · the first counted grade

**NULL.** On the 904 midday draws of the grade block, arm A1 (the frozen CatBoost ranking) hit
**22 of 904**. Chance at its own k_t predicts **17.91**. One-sided p is **0.1925** with m = 1, so
the result is **not above chance at 5 %**. All controls passed. No descriptive run was above
chance either.

This is a measured null for one declared configuration. It closes no lead and it is no verdict on
TFM (limits in §5).

## Prior record

**Searches run:**
- Command: `/bin/grep -rlniE "e2e_runH_grade|counted grade|RESULT_E2E_GRADE"`.
- Scope: `docs/PROJECT_FILE_CATALOG.md`, `docs/TB_RULING_*`, `docs/PROPOSAL_*`, `docs/CHAPTER_*`,
  `docs/*_AUDIT.md`, `docs/TODO_CONSOLIDATED.md`, `docs/BACKLOG.md`, `docs/LEADS.md`,
  `docs/exchange/` and `logs/e2e_session/`, all in the main tree at `41ad509`.
- Also searched: `~/tfm_worktrees/*/docs/drafts/e2e_v1/` for any earlier RESULT file.

**Found:**
- The task file `logs/e2e_session/TASK_GRADE_RUNH.txt`.
- The owner go `logs/e2e_session/OWNER_GO_counted_grade_20261123.txt` (13:44).
- `TASK_COMMON.txt`, which mentions a counted grade only as needing the owner's go.
- Authority for the declaration:
  - `OWNER_RULING_e2e_Q1_20261123.txt` (08:08): label block / grade block cut, midday counted
    (904), one counted test, one-sided 5 %;
  - `OWNER_RULING_step5_go_ch14_beta_holds_20261123.txt` (10:40).
- No earlier grade of run H pools and no earlier RESULT file exists.

**Labels:**
- The counted test and its block: **ALREADY RECORDED** (Q1 ruling; TASK_GRADE_RUNH STEP 1).
- The hit counts below: **NEW** (first grade).
- **CHANGES A RULING:** none.

**Not searched:** the host, ser8 archives, the private repository, project files outside the
repository.

## 1. What was declared, and when (STEP 1)

The declarations were written and frozen before any outcome was read: read-only files, sha256s
in `data/e2e_runH_grade/declared/SHA256SUMS`, all re-verified OK after the runs.
- Time written: **2026-11-23 13:50:34 PST** (`declared/run_record_grade.json`, sha256
  `4faedf8bc01706f72a3b7bc411c9437dd84efb8a0fdcd63536e838ca08724f0b`).
- The counted run's summary is stamped 13:50:48.
- Generator: `data/e2e_runH_grade/declare.py`. It read the pool files' bytes and `analysis.json`,
  never a draw value.

**Dataset check, before the run:**
- `daily3.json` has 18,068 rows. Every row has exactly the keys {date, draw, session}, and `draw`
  is an int. This matches the grader's reader (`e2e_grader.py:233-257`).
- I checked keys and types only; no value was looked at.
- Grade block [16261, 18068): 904 midday and 903 evening rows.

| File | sha256 | Content |
|---|---|---|
| `grade_block_midday.json` | `8466e0a3…8196` | [16261, 18068), sessions [midday]. Frequency block [0, 14454) midday |
| `grade_block_evening.json` | `29024df0…7e3877` | Same range, [evening]. Frequency block [0, 14454) evening (D3) |
| `arms_counted_A1.json` | `3ca5fa33…5e13` | A1 only: config_id `bd932b3e…68bc`, upstream_unseen_from_index 16261, min_k 1, max_k 20, no comparisons, family_alpha 0.05 |
| `arms_D1_A0.json` | `c743ef97…f0af` | A0 only: config_id `52cbe211…b2f1`, upstream 14454 |
| `arms_D2_A1_A0.json` | `69f16e61…a874` | A1 + A0, comparison A1 vs A0 |
| `arms_D3_A1.json`, `arms_D3_A0.json` | same bytes as the counted A1 and D1 A0 files | |
| `abstention.json` | `0d0471c9…9e50` | allow_abstention false, empty_pool error |
| `pools_A1_A0_concat.jsonl` | `b1c541d3a490d9b7004a26b0789e9655b6f774e3fdb251b1f98517cc1f08fe17` | A1 file then A0 file, byte concatenation (D2) |

**Flags:** `--out-of-block-records exclude_and_count`. Everything else is the grader default:
- 200 control replicates, tolerance 4 SE, seed 20261123;
- 4 workers;
- Bonferroni over `all_declared_tests`.

**Inputs, verified:**
- The pool sha256s match `~/tfm_worktrees/e2e-vote/data/e2e_runH_vote/analysis.json`:
  - A1 `full_A1/pools.jsonl` is `d0a80f8eca7031e9470dbbc5113c4d16b3b29fcfc74029a8d5a8ec48a340aa12`;
  - A0 `full_A0/pools.jsonl` is `7c1cc332bf29daee4cf751339a6b4b344c147bbd23a57faba292364e44014506`.
- Dataset `/home/michael/skipsw_run_thr3030/daily3.json` is
  `513648160d356617c22a1e543ae1c9c65f4921ec21718989308b1f70c00768f6`.
- Grader `scripts/diagnostics/e2e_grader.py` is
  `a4d96fe7ac441a8befc93f55a638eb70e36e7d0cd70f24eaeb60d08b7a52169a`. It was not edited, and its
  sha256 was re-checked after the runs.
- Provenance recorded by the grader:
  - commit `41ad509`;
  - `tree_dirty: true` (the grader and tests are still untracked);
  - Python 3.10.12, numpy 1.22.0, scipy 1.11.4.

## 2. The counted test: A1 vs chance, midday, m = 1

The grader exited **0** with `controls_ok=True`. Its headline, verbatim
(`data/e2e_runH_grade/counted_A1_midday.log`):

> A1: 22 hits of 904 graded draws (expected 17.91 at k/1000; abstained 0 of 904 eligible; pending 0); one-sided p 0.1925, adjusted p 0.1925 (m=1, bonferroni); lift lower bound 0.835; NULL: not above chance after correction

**Result** (`counted_A1_midday/summary.json`):

| Quantity | Value |
|---|---|
| Graded / eligible | 904 / 904 midday draws; abstained 0; pending 0 |
| Excluded out of block (evening, written, not counted) | 903 of 1,807 records |
| Distinct k_t | 18 on 15 draws, 19 on 143, 20 on 746; Σk_t = 17,907 |
| Hits | **22** of 904 (hit rate 0.02434) |
| Expected hits Σk_t/1000 | **17.907** (baseline rate 0.01981) |
| Null sd | 4.1895 |
| One-sided p = P(X ≥ 22) | **0.19247** |
| Adjusted p (m = 1) | 0.19247; above_chance **false** |
| Lift point | 1.2286 |
| Lift one-sided lower bound at α = 0.05 | 0.8346 |
| Lift two-sided 95 % interval (descriptive) | [0.7730, 1.8484] |
| Lag-1 autocorrelation of residuals | −0.0250 |

**First count that would have reached p ≤ 0.05 for A1's actual k_t values: 26 hits.**
- P(X ≥ 26) = 0.04084 and P(X ≥ 25) = 0.06332.
- Computed with the grader's own `pb_upper_tail` on the 904 graded k_t (`extract.py`, output
  `extract.json`).
- 22 is 4 hits short of that count.

**Controls.** All passed (`controls_ok=True`).

| Control | Numbers | Verdict |
|---|---|---|
| Uniform-random pools, same k_t | 200 replicates. Mean 17.995 hits against expected 17.907; SE 0.296; z = 0.297 (tolerance 4). Critical count 26; exact size 0.0408; empirical rejection rate 0.040 | pass |
| Known-count | 452 hits against 452 expected on 904 draws | pass |
| Draw-frequency (descriptive, not in m) | The 20 most frequent midday values in [0, 14454), from 6,707 rows. **16 hits** against 17.907 expected on 904 draws; p 0.7083 unadjusted | reported |

## 3. Descriptive runs: DESCRIPTIVE, NOT COUNTED

Each ran once, in its own new out-dir, and exited 0 with `controls_ok=True`. None is above chance.
The headlines are quoted verbatim from each `.log`.

**D1. A0 alone, midday, upstream 14454** (`D1_A0_midday/`)
> A0: 21 hits of 904 graded draws (expected 17.91 at k/1000; abstained 0 of 904 eligible; pending 0); one-sided p 0.2607, adjusted p 0.2607 (m=1, bonferroni); lift lower bound 0.788; NULL: not above chance after correction

| Quantity | Value |
|---|---|
| Distinct k_t | 18 on 11 draws, 19 on 145, 20 on 748; Σk_t 17,913 |
| Null sd | 4.1902 |
| Lift point | 1.1723 |
| 95 % interval (descriptive) | [0.7285, 1.7810] |
| First count reaching p ≤ 0.05 | 26 |
| Uniform control | mean 18.04, z 0.429, rejection rate 0.045 against size 0.0410 |
| Known-count | 452/452 |
| Frequency control | 16 of 904 |

**D2. A1 and A0 together, midday, comparison A1 vs A0** (m = 3; `D2_A1_A0_midday/`)
> A0: 21 hits of 904 graded draws (expected 17.91 at k/1000; abstained 0 of 904 eligible; pending 0); one-sided p 0.2607, adjusted p 0.7821 (m=3, bonferroni); lift lower bound 0.698; NULL: not above chance after correction
> A1: 22 hits of 904 graded draws (expected 17.91 at k/1000; abstained 0 of 904 eligible; pending 0); one-sided p 0.1925, adjusted p 0.5774 (m=3, bonferroni); lift lower bound 0.742; NULL: not above chance after correction

**Comparison A1 vs A0:** **not tested: unequal budget** (k_t differs on some common draw).
- 904 common draws.
- Hits 22 vs 21; expected 17.907 vs 17.913; excess over expected +4.093 vs +3.087.
- Discordant draws: 21 hit by A1 only, 20 by A0 only.
- Uniform controls at α/3: critical count 28, size about 0.0154, rejection rate 0.010 for each arm.
  Both pass.

**D3. Evening, frequency block evening [0, 14454) from 7,747 rows** (`D3_A1_evening/`, `D3_A0_evening/`)
> A1: 16 hits of 903 graded draws (expected 17.89 at k/1000; abstained 0 of 903 eligible; pending 0); one-sided p 0.7072, adjusted p 0.7072 (m=1, bonferroni); lift lower bound 0.562; NULL: not above chance after correction
> A0: 22 hits of 903 graded draws (expected 17.89 at k/1000; abstained 0 of 903 eligible; pending 0); one-sided p 0.1917, adjusted p 0.1917 (m=1, bonferroni); lift lower bound 0.835; NULL: not above chance after correction

| | A1 | A0 |
|---|---|---|
| Lift point | 0.8942 | 1.2294 |
| 95 % interval (descriptive) | [0.5128, 1.4441] | [0.7736, 1.8496] |
| First count reaching p ≤ 0.05 | 26 | 26 |
| Uniform control | z 0.176, rejection 0.050 against size 0.0405 | z 0.034, rejection 0.040 against size 0.0406 |
| Known-count | 452/452 | 452/452 |
| Frequency control | 21 of 903, p 0.2591 | 21 of 903, p 0.2593 |

The 904 midday records are excluded and counted in each D3 run.

**Note on the uniform controls.** The replicate series for A1 and A0 are almost the same, because
both use seed 20261123 per replicate and nearly the same k_t. They are not independent checks of
each other.

## 4. Run facts

| Run | Exit | Wall | Peak RSS |
|---|---|---|---|
| counted_A1_midday | 0 | 3.6 s, 299 % CPU | 96 MB |
| D1_A0_midday | 0 | 3.6 s | 96 MB |
| D2_A1_A0_midday | 0 | 6.7 s, 315 % CPU | 106 MB |
| D3_A1_evening | 0 | 3.6 s | 96 MB |
| D3_A0_evening | 0 | 3.6 s | 96 MB |

- Every run was on CPU, multi-core (4 workers for the uniform control), and ran once.
- No other run, k or setting was used.
- Nothing was written outside `data/e2e_runH_grade/` except this note.

**Output sha256s** (from each out-dir's `SHA256SUMS`):

| Out-dir | `per_draw.jsonl` | `summary.json` | `SUMMARY.md` |
|---|---|---|---|
| counted_A1_midday | `98ccf7bded7694ce6d4644ee68661a31fc5bce603b067d78f333061f152dc0c7` | `bf15a4a60d30c6efc9da8cebf28e7ff77c8fa2cef8067a8ff84ca479efd42e1d` | `92366a3da46b9eb4a356f960c0819cbf5c00816557d562fa4f57fcd776dd443d` |
| D1_A0_midday | `52db19ad…3615` | `c6a48771…beca` | `2c472366…f283` |
| D2_A1_A0_midday | `644e2a25…91b0f8` | `cb32141…caea` | `caef1518…d54f` |
| D3_A1_evening | `400abce9…7af5` | `0fe306f8…91c1` | `5399b323…2c7c` |
| D3_A0_evening | `a9e4a123…87a` | `a0e00e1f…bf` | `9663ae4b…65` |

The full hashes are in each `SHA256SUMS`.

**Working files:**

| File | sha256 |
|---|---|
| `declare.py` | `75aac46d…6c36` |
| `extract.py` | `8bb0fec6…c36` |
| `extract.json` | `4544c1c9…b019` |

## 5. Limits, stated plainly

These limits are SOURCE: `logs/e2e_session/TASK_GRADE_RUNH.txt`, plus the vote analysis where
cited.
- **One generation.** Run H only:
  - 12,156 survivors (`analysis.json` ranking `rows` 12156);
  - a sieve window of 12 midday draws at offset 25, dated 2002;
  - thresholds 0.30/0.30;
  - 1/16 of the seed space;
  - one trial.
- **Step 6 vote convention as built.** Combined index, effective skip 0 for all 12,156 survivors
  (`analysis.json` `effective_skip_counts`). The combined order puts evening before midday from
  row 1039 (READY note Q4); this grade is per session.
- **Features and target as built.**
- **A frozen CatBoost ranking.** Its cross-validation score was +0.0019. It was ranked once, and the
  top-20 survivor set was constant across draws (`analysis.json` `ranking_once`, `top20_constant`).
- **Instrument status.**
  - Non-certifying harness route.
  - The grader is a draft, WP-3 first form, not Beta-reviewed. The lead authorised it unchanged
    for this grade.
- **Power.** With 904 draws at k about 20, this test needs 26 hits, about 1.45× chance, to reach
  5 %. A smaller real lift could exist and not be detected here.

## 6. Reading

- This is a measured null for this declared configuration: run H, A1, midday grade block, k ≤ 20,
  904 draws. **It closes no lead.**
- No descriptive run was above chance, and those runs carry no counted weight.
- **No verdict on TFM either way.**
