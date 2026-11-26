# NOTE: lead L-8, the two-way overlap in runs E–H exceeds the independence figure (2026-11-22)

**Summary, in plain terms (12 lines):**
1. Every number the lead gave for runs E and F checks out against the files. Run H's phase 3 and 4 figures are the exception: they add in the constant-mode rows (§1).
2. A CPU port of the repository's reference sieve reproduces runs E, F, G and H exactly: 34,454 / 48,791 / 17; 34,419 / 48,449 / 13; 326,627 / 48,791 / 167; 2,368,585 / 595,773 / 12,156.
3. The independence figure (fwd × rev / 2^28) is an assumption. The measured two-way count is 1.8× to 3.0× above it at every threshold pair on the real window.
4. The owner's question: is this overlap a property of the sieve arithmetic alone, or does it depend on the draws?
5. Measured answer (CPU, 351 windows × 2^24 seeds): it appears whenever both passes compare against the same 12 values, whatever they are.
6. Real windows at 100 other anchors, 100 shuffles of this window and 100 windows of random values all give the same excess (2.35–2.40× at 4-of-12 each way).
7. When forward and reverse are given two unrelated windows, the excess goes away (1.01–1.13×).
8. Mechanism: both passes walk one output stream from the same seed with wide skip windows. In 26 of the 30 published rows, at least one output counts as a hit for the same draw in both passes.
9. The anchor-25 window's excess (2.41× at the run thresholds) is not above the control groups' (3.3–3.9×) at the same thresholds.
10. Neighbourhood, as a description of the population only: the 30 rows are spread across both slices. They share no residue class mod 1000, 2^16 or 2^19, none follows from another within 2^20 generator steps, and survivor density around them is at the slice average.
11. For hit rate: the excess carries no information about pool hit rate. Only pools scored on unseen draws against `k/1000` settle value (§6).
12. The lead stays **OPEN**; downgrading it is the operator's decision. The main line, a baseline population handed to Step 2, is not gated by it.

- **Author and scope:** Claude (Team Alpha), worktree `/home/michael/tfm_worktrees/pwc-offset`, uncommitted. CPU only: no GPU, no rig, no commit.
- **Binding:** owner 14:08, "do not fall back into the reversing state mentality", and `docs/PITFALL_STATE_RECOVERY_YARDSTICK.md`.
- **What the seeds are here:** vehicles that passed a filter. No line asks or implies that any one of them is "the" seed.

## Prior record

**Searches run**, on the main tree at `41ad509`, with `/bin/grep -rl -E -i` over `docs/PROJECT_FILE_CATALOG.md docs/TB_RULING_* docs/TB_RULING_REQUEST_* docs/PROPOSAL_* docs/CHAPTER_* docs/*_AUDIT.md docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/LEADS.md docs/exchange/`:

| # | Pattern | Hits read |
|---|---|---|
| S1 | `bidirectional excess\|independence\|P\(survive both\)\|forward.*reverse.*correlat\|correlat.*forward.*reverse` | 10 files. Read: `CHAPTER_2_BIDIRECTIONAL_SIEVE.md:318-330`; `exchange/from_beta/TFM_Reverse_Sieve_Proposal_2026-11-13.md:18`, `:100` |
| S2 | `shuffled\|shuffle.*draw\|permut.*draw\|synthetic.*draw\|control arm\|sampler_control` | 25+ files, mostly Chapter 13. Read: `tests/phase6/sampler_control_arm.py:1-60` (an optimizer sampler control, not a draw control) |
| S3 | `dataset authority\|authoritative dataset` (and `dataset.{0,40}authorit`) | Read: `BACKLOG.md:104-120` §5; `miner/dataset_authority.py:1-66`, `:477-583`, `:1248-1300`; `utils/seed_coverage_ledger.py:862-905` |
| S4 | `docs/LEADS.md` headings | L-1 to L-7. None covers the sieve's two-way count |

**Documents that already record the subject:**
- **Chapter 2, open item F-7 / PIPELINE_BEHAVIOUR_MODEL §16 D10** (`CHAPTER_2_BIDIRECTIONAL_SIEVE.md:318-330`). For one seed, the forward and reverse passes generate the identical output sequence and differ only in what it is compared against. The whitepaper's independence premise (§4 `:61-62`, the squaring at §5 `:79`) is stated about a different construction. The chapter "does not assert the statistical consequence either way".
- **Beta, 2026-11-13** (`exchange/from_beta/TFM_Reverse_Sieve_Proposal_2026-11-13.md:18`, `:100`): "reversing the order of the same comparisons does not make them statistically independent"; "`P(F and R) = P(F) * P(R given F)`. Substituting `P(R)` for the conditional probability needs support."
- **Lead-handling rule** (`~/.claude/skills/lead-handling/SKILL.md` §2, §5): a background figure must state which inputs were measured and which assumed; a lead closes only on a failed follow-up.

**Labels:**

| Item | Label |
|---|---|
| The two passes are not independent | **ALREADY RECORDED** as a question (F-7, D10, Beta :100). The **measurement** of `P(R∣F)/P(R)` below is **NEW** |
| Lead entry L-8 | **NEW** |
| Rig control scope (§5) | **NEW**. Touches `BACKLOG.md` §5 (session-separated authority, ALREADY RECORDED) |
| Anything that changes a ruling | **none** |

**Not searched:** the host outside the paths named here, the ser8 archives, the private repository, project files that are not in the repository, the changelog corpus and the `apply_s*.py` corpus.

## 1. The lead's numbers, verified from the files

| Lead's figure | File | Read |
|---|---|---|
| Ladder (VM101), window 12, anchor 25, midday, skip [0,16] | `~/skipsw_survey/ladder.out`, `ladder2.out`, `params_*.json`, `run_ladder2.sh` | At τ 0.47, phase 3 has 8,665 hits of 67,108,864 sampled (projection 34,660). At τ 0.38, phase 4 has 48,791 of 268,435,456. The interval is [2147483648, 2415919104), the same as run E |
| E tree `~/skipsw_run_thr4738`; fwd 34,454, rev 48,791, 17 rows | `logs/skipsw_session/bundle_thr4738/bundle.json` `phase_counts`, `inputs`; `survivors.npz` | Phase 3 has 34,454 rows and phase 4 has 48,791; phases 1 and 2 have 0. τ_f 0.47, τ_r 0.38; window 12, anchor 25, session index 1; switch on; commit `00e2ee7`. 17 rows, all `skip_mode` 1. Credit granted |
| F on [2415919104, 2684354560), tree `~/skipsw_run_thr4738n`; 34,419 / 48,449 / 13 | `bundle_thr4738n` | Matches. 13 rows, all variable |
| Receipts | `receipt_thr4738{,n}.json` | Hashes recompute to the recorded `.sha256` values (`e1c135e2…`, `938f3839…`). Run copies pass `sha256sum -c`: 94/94 each, 0 failures |
| Blocks `~/skipsw_blocks/e`, `f` | `b01.sh`–`b18.sh`, `runE_post.log`, `runF_post.log` | Present. Post logs: switch lines, credit, `search_complete`, rig check "no workers" ×3, receipt and bundle lines |
| 6.26 and 6.21 expected | arithmetic | 34,454 × 48,791 / 2^28 = **6.262**; 34,419 × 48,449 / 2^28 = **6.212**. Both rest on an **assumption** of independence |
| G (0.38/0.38, E slice): 326,627 / 48,791 / 167, about 59 | `bundle_thr3838` | Matches. 167 rows, all variable. Independence figure 59.37, ratio 2.81 |
| H (0.30/0.30, E slice): "phase3 2,368,586, phase4 595,775", 12,156, about 5,257, ratio 2.3 | `bundle_thr3030`, `receipt_thr3030.json` (`4505e887…`, matches its `.sha256`) | **Phase 3 = 2,368,585; phase 4 = 595,773.** Phase 1 (constant forward) = 1, phase 2 (constant reverse) = 2. The lead's figures are phase 1 + 3 and phase 2 + 4. Bidirectional = 12,156 rows, all `skip_mode` 1 (variable). Independence figure 2,368,585 × 595,773 / 2^28 = 5,256.9; ratio 2.312 |

**Constant mode found nothing two-way at any of these thresholds.** Every published row is variable-skip (phases 3 ∩ 4).

## 2. Test the mechanism: arithmetic alone, or the draws?

### 2.1 What the kernels do (read from `prng_registry.py` at `41ad509`)

**Forward hybrid,** `java_lcg_hybrid_multi_strategy_sieve` (`:1007`):
- It starts from the raw seed (`state = seed & m`, `:1024`), with no generator-phase argument.
- `expected_skip = 5` (`:1027`). Each draw searches skips `[expected − tol, expected + tol]` (`:1033-1034`) and re-centres on a hit (`:1050`).
- On a miss, the state is left advanced by `search_max` steps.
- It breaks after `max_misses` consecutive misses (`:1057`) and keeps the strategy with the strictly best rate (`:1061`).

**Reverse hybrid,** `java_lcg_hybrid_reverse_sieve` (`:3172`):
- It also starts from the raw seed, then pre-advances by the trailing argument (`:3190`); that is `generator_phase` = 0 in all four runs, per `logs/stage4/volume_params.json`.
- Each draw searches skips 0..tol from the current position (`:3200`), and restores the state on a miss (`:3218`).
- A strategy fails after more than `max` consecutive misses (`:3223`). The kernel emits the first non-failed strategy that clears τ (`:3229-3239`).
- **The direction comes from the host:** `residues[::-1] if reverse else residues` (`miner/range_miner_worker.py:1330`).

**Strategies.** The coordinator sends no `strategies`, so the worker uses all five presets of `hybrid_strategy.STRATEGY_PRESETS` (`miner/range_miner_worker.py:1227-1238`; `hybrid_strategy.py:33-79`). They include `extreme_tolerance`: tol ±50, 20 misses, so it never breaks on a 12-draw window.

**Reading, as it bears on the owner's question.** Both passes iterate one output stream from the same seed. Each covers up to about 100 (forward) or 51 (reverse) stream positions per draw, so for most seeds the two walks pass over the same stretch of stream early on. An output equal to draw `d`'s value can therefore count as a hit in both passes. That is a shared-input mechanism, and it needs no structure in the draws. Taking the lead's three questions in turn:
- **Shared residues:** both passes use the same 12 values; this is the mechanism above.
- **Palindromic or repeated draws:** the run window `[413, 437, 707, 504, 670, 383, 330, 221, 505, 142, 335, 307]` has no repeated value and no mirror-equal pair.
- **Same skip search:** no. The two kernels search skips differently (re-centred ±tol against 0..tol).

### 2.2 The CPU measurement

**Instrument.**
- `logs/lead_bidir_20261122/hyb.py` (sha256 `1ed11385…`) is a numba port of the repository CPU reference `tests/phase6/known_answer_reference.py` (`369bf5c4…`): `hybrid_forward_sieve` `:234`, `hybrid_reverse_sieve` `:323`.
- It returns integer match counts F and R per seed, so any threshold can be applied afterwards.

**Validation** (`validate.py`, `validate.log`):
- 1,500 random seeds at every threshold m/12, m = 0..8: the forward and reverse sets equal the reference's.
- All 30 published seeds pass both passes in the port and in the reference.
- **Full-interval reproduction** (`reproduce_EF.py`, `.log`, `.json`), over 2^28 seeds each:
  - E: 34,454 / 48,791 / 17;
  - F: 34,419 / 48,449 / 13;
  - the 30 seeds are identical to the two NPZs;
  - G's and H's counts fall out of the same histogram (E slice).
- **Measured, not assumed:** the CPU path equals the rig path on four runs.

**Real window, full intervals** (2^29 seeds; `analysis.json` `real_full_intervals`). The ratio is `P(both) / (P(fwd)·P(rev))`; the 95% intervals are Garwood intervals on the both-count.

| Thresholds (matches of 12) | E+F both | Independence figure | Ratio [95%] |
|---|---|---|---|
| F ≥ 6, R ≥ 5 (runs E, F) | 30 | 12.47 | **2.41** [1.62, 3.43] |
| F ≥ 5, R ≥ 5 (run G) | 327 | 118.1 | 2.77 [2.48, 3.09] |
| F ≥ 6, R ≥ 4 | 454 | 152.9 | 2.97 [2.70, 3.26] |
| F ≥ 5, R ≥ 4 | 3,786 | 1,447.6 | 2.62 [2.53, 2.70] |
| F ≥ 4, R ≥ 4 (run H) | 24,414 | 10,515 | 2.32 [2.29, 2.35] |
| F ≥ 4, R ≥ 3 | 189,515 | 92,882 | 2.04 [2.03, 2.05] |
| F ≥ 3, R ≥ 3 | 949,193 | 524,527 | 1.81 [1.81, 1.81] |

**Controls** (`controls.py` `ce8a9b83…`, `controls.json`, `analysis_controls.log`). Each window gets the same 2^24 random seeds from the union of the E and F slices (numpy `default_rng(1122)`). Pooled ratio = Σ both / Σ independence figure, with a Poisson interval and a 4,000-draw bootstrap over windows.

| Group | Windows | What differs from the run | F≥6 R≥5 ratio [Poisson 95%] | F≥4 R≥4 ratio [95%] | F≥3 R≥3 |
|---|---|---|---|---|---|
| other | 100 | real midday windows at random other anchors | 3.47 [2.91, 4.11] | 2.40 [2.38, 2.41] | 1.84 |
| shuffled | 100 | the run window's 12 values, permuted | 3.91 [3.31, 4.58] | 2.35 [2.33, 2.36] | 1.81 |
| iid | 100 | 12 random values 0–999 | 3.32 [2.77, 3.94] | 2.37 [2.36, 2.39] | 1.82 |
| **cross** | 50 | **forward on window A, reverse on an unrelated window B** | **1.13** [0.71, 1.72] | **1.05** [1.03, 1.06] | **1.01** |
| real (sampled) | 1 | the run window, the same 2^24 seeds | 0 both, independence figure 0.39 | 2.39 [2.23, 2.57] | 1.81 |

**Per-window spread at F≥4 R≥4** (min, Q1, median, Q3, max):

| Group | min | Q1 | median | Q3 | max |
|---|---|---|---|---|---|
| other | 2.14 | 2.30 | 2.34 | 2.40 | 4.97 |
| shuffled | 2.16 | 2.27 | 2.35 | 2.41 | 2.54 |
| iid | 2.15 | 2.32 | 2.37 | 2.43 | 2.81 |
| cross | 0.90 | 0.97 | 1.01 | 1.04 | 2.31 |

The anchor-25 window on the full intervals gives 2.32.

**Windows with a repeated value or a mirror-equal pair run higher** (the `other` group). With no repeated value, 93 windows have a median of 2.34; with one repeat, 7 windows have a median of 2.50 (one at 4.97). The 2 windows with a mirror-equal pair have a median of 3.71.

**Answer to the owner's question, as measured:**
- The two-way excess is **a property of the sieve arithmetic when both passes are compared against the same values.** It appears at the same size for real, shuffled and random windows, and disappears when the two passes are given different values.
- **It does not depend on which draws, their order, or whether they are real.** It rises with repeated or mirror-equal values within a window, which the run window has none of.
- **At the run thresholds the anchor-25 window sits below the control groups:**
  - Using the iid ratio (3.32, measured on other values), the figure for E+F is 12.47 × 3.32 ≈ **41.4 [34.5, 49.1]**.
  - **Observed: 30.**
  - **Measured inputs:** the marginals (this window, rig and CPU) and the ratio (iid controls).
  - **Assumed:** that the iid ratio transfers to this window. The F≥4 R≥4 per-window spread supports that, but the assumption is not proven at F≥6 R≥5.

**Disclosure required by the lead-handling rule:**
- The 6.26 and 6.21 figures assume independence; that input is now **measured to be false** (ratio 1.8–3.9 across thresholds).
- The control figures above are **measured** on CPU with a port validated against the rig.
- **The licensed phrasing:** "Its rate is consistent with the measured same-window background (controls in §2.2). No follow-up has tested whether it extends or recurs as a hit-rate effect. It remains an open lead."

## 3. Neighbourhood: a description of the population only

**The 30 published rows** (`neighbourhood.py` `d8a31b9b…`, `.json`, `.log`; computed with the repository reference, not the port):
- **Matches:** F and R are counts of 12.
- **Skip columns:** the kernels' per-draw skip sequences. The forward sequence holds `expected_skip` on a miss; the reverse holds `try_skip` on a hit and 0 on a miss. The NPZ does not keep them: its `skip_min`/`skip_max` columns are the configured [0,16].
- **"Both" column:** the draw indices (0-based, draw order) matched in both passes, and how many stream outputs count as a hit in both.
- **Agreement:** the NPZ `forward_matches`/`reverse_matches` agree with the reference for all 30.

| Run | Seed | F | Forward strategy | Forward skips | R | Reverse strategy | Reverse skips | Draws hit in both | Shared outputs |
|---|---|---|---|---|---|---|---|---|---|
| E | 2170280596 | 6 | extreme_tolerance | 5 35 35 71 116 116 101 136 136 136 137 137 | 5 | extreme_tolerance | 0 0 0 0 38 0 0 12 0 23 14 39 | 1,4,7 | 1 |
| E | 2190388066 | 6 | extreme_tolerance | 5 44 1 1 16 40 40 40 40 40 87 55 | 5 | extreme_tolerance | 0 6 0 0 12 3 12 0 0 0 0 22 | 5,10 | 0 |
| E | 2208546727 | 6 | extreme_tolerance | 8 40 40 40 25 25 25 72 72 43 11 11 | 5 | extreme_tolerance | 0 3 0 0 0 0 0 17 0 0 26 16 | 0,1,4,7,10 | 1 |
| E | 2214258821 | 6 | extreme_tolerance | 27 75 75 57 57 57 57 48 48 56 84 84 | 5 | extreme_tolerance | 0 0 41 0 22 0 0 0 0 6 31 0 | 0,1,7,9 | 1 |
| E | 2226445142 | 6 | extreme_tolerance | 1 37 37 61 38 38 83 83 83 83 60 60 | 5 | extreme_tolerance | 7 0 47 0 0 37 0 47 46 0 0 0 | 3,4,6 | 1 |
| E | 2232281722 | 6 | extreme_tolerance | 5 5 40 40 90 72 72 33 76 76 106 106 | 5 | extreme_tolerance | 0 0 0 0 21 43 0 24 36 22 0 0 | 2,4,7 | 1 |
| E | 2237975527 | 6 | extreme_tolerance | 5 45 45 88 86 86 37 37 16 16 10 10 | 5 | extreme_tolerance | 0 0 0 11 0 5 0 0 10 26 44 0 | 1,3,6,8 | 1 |
| E | 2239518718 | 6 | extreme_tolerance | 40 40 40 15 15 63 87 117 137 137 137 137 | 5 | lenient_continuous | 0 0 5 4 10 0 0 0 0 16 0 1 | 0,7,8 | 1 |
| E | 2254168162 | 7 | extreme_tolerance | 5 45 77 77 77 92 92 114 74 72 72 119 | 5 | extreme_tolerance | 0 0 0 33 0 0 13 7 0 0 44 36 | 1,5,8 | 1 |
| E | 2255567924 | 6 | extreme_tolerance | 16 1 1 30 30 30 19 36 36 13 13 13 | 5 | extreme_tolerance | 0 36 0 0 5 0 0 23 33 0 32 0 | 1,3,7 | 1 |
| E | 2260411044 | 6 | extreme_tolerance | 5 5 29 29 68 68 50 94 108 108 108 135 | 5 | extreme_tolerance | 0 40 14 0 0 0 46 0 0 36 6 0 | 2 | 1 |
| E | 2276183193 | 6 | extreme_tolerance | 5 5 0 20 2 2 2 39 39 32 18 18 | 5 | extreme_tolerance | 0 0 0 11 42 31 0 47 29 0 0 0 | 3,4,7 | 1 |
| E | 2284428144 | 6 | extreme_tolerance | 5 43 43 43 43 78 110 123 106 138 138 138 | 5 | extreme_tolerance | 0 26 0 0 0 0 27 0 0 18 24 30 | 1,5 | 1 |
| E | 2372128352 | 6 | extreme_tolerance | 24 57 57 57 26 26 26 26 53 53 23 23 | 5 | extreme_tolerance | 0 25 0 0 0 0 0 24 14 14 1 0 | 1,2,4,10 | 1 |
| E | 2387303058 | 6 | extreme_tolerance | 5 34 83 83 83 88 108 108 108 108 104 149 | 5 | extreme_tolerance | 41 0 0 0 9 8 0 26 0 0 1 0 | 1,6,11 | 1 |
| E | 2397052730 | 6 | extreme_tolerance | 18 9 40 45 45 73 73 73 73 101 101 101 | 5 | extreme_tolerance | 41 0 0 0 0 11 29 0 31 22 0 0 | 2,3,5 | 1 |
| E | 2404401812 | 6 | extreme_tolerance | 5 29 29 68 104 104 104 104 72 25 32 32 | 5 | extreme_tolerance | 0 0 0 16 0 14 0 24 0 0 27 8 | 1,4,8 | 1 |
| F | 2421316199 | 6 | extreme_tolerance | 5 5 24 64 106 106 102 102 102 117 117 163 | 6 | extreme_tolerance | 24 42 0 42 0 47 0 0 40 0 28 0 | 3,6,11 | 1 |
| F | 2428128897 | 6 | extreme_tolerance | 5 2 32 57 57 57 57 57 93 79 79 103 | 6 | extreme_tolerance | 43 0 0 0 0 33 0 32 37 6 44 0 | 1,2,3,11 | 1 |
| F | 2443997463 | 6 | extreme_tolerance | 5 13 53 76 76 76 76 76 114 111 108 108 | 5 | extreme_tolerance | 35 0 1 5 0 0 23 0 0 0 0 0 | 1,8,9 | 1 |
| F | 2465026487 | 6 | extreme_tolerance | 5 5 46 58 58 58 58 73 101 101 72 23 | 5 | extreme_tolerance | 0 0 18 0 0 0 45 0 18 21 0 14 | 2,3 | 0 |
| F | 2467879549 | 6 | extreme_tolerance | 51 23 23 23 24 24 70 112 113 113 113 113 | 5 | extreme_tolerance | 3 0 37 0 28 0 0 0 0 0 4 12 | 0,1,7 | 1 |
| F | 2473828036 | 6 | extreme_tolerance | 55 55 36 36 36 42 42 42 89 111 111 139 | 5 | extreme_tolerance | 2 0 0 6 0 10 9 0 0 0 0 24 | 0,5,8,11 | 1 |
| F | 2523885616 | 7 | extreme_tolerance | 5 5 5 40 65 65 68 68 24 70 36 19 | 5 | extreme_tolerance | 0 20 38 0 19 49 0 0 0 0 25 0 | 6,9,10 | 0 |
| F | 2532248589 | 6 | extreme_tolerance | 5 5 42 42 45 45 79 103 103 85 85 96 | 5 | extreme_tolerance | 0 35 0 0 0 0 36 0 45 33 14 0 | 2 | 1 |
| F | 2537344852 | 6 | extreme_tolerance | 42 72 72 72 72 46 87 87 65 63 63 63 | 5 | extreme_tolerance | 41 0 13 0 38 15 0 0 0 0 4 0 | 1,6,9 | 1 |
| F | 2592240531 | 6 | extreme_tolerance | 27 74 74 65 86 36 36 36 36 36 2 2 | 5 | extreme_tolerance | 0 44 37 0 0 0 0 0 11 0 7 34 | 0,1,3,10 | 1 |
| F | 2610079307 | 6 | extreme_tolerance | 13 39 39 39 78 85 85 85 85 92 71 71 | 5 | extreme_tolerance | 0 33 10 0 49 0 0 0 7 0 0 36 | 0,9,10 | 0 |
| F | 2628527173 | 6 | extreme_tolerance | 5 22 58 51 51 98 98 98 98 98 134 149 | 5 | extreme_tolerance | 1 38 0 0 0 22 0 23 0 48 0 0 | 2,10,11 | 1 |
| F | 2676333020 | 6 | extreme_tolerance | 5 5 29 61 24 43 43 43 73 118 118 118 | 5 | extreme_tolerance | 0 41 16 0 0 0 34 38 0 6 0 0 | 2,4,5,9 | 1 |

**Population description** (`neighbourhood.json`, `density.log`):
- **Match counts:** 28 rows at the minimum forward count (6 of 12) and 2 at 7; 28 at the minimum reverse count (5) and 2 at 6.
- **Strategies:** forward winner `extreme_tolerance` in 30/30; reverse first-qualifying strategy `extreme_tolerance` 29/30, `lenient_continuous` 1/30.
- **Shared outputs:** in 26 of 30 rows, one stream output counts as a hit for the same draw in both passes (the §2.1 mechanism).
- **Spacing:** sorted gaps run from 1,399,762 to 87,700,208, median 12,186,321, coefficient of variation 1.09. Uniform placement would give about 1.0 [assumed reference].
- **Residue classes:** odd/even 17/13. Mod 8: {0:3, 1:2, 2:5, 3:2, 4:7, 5:4, 6:2, 7:5}. No two rows share a value mod 1000, mod 2^16 or mod 2^19.
- **Low bits:** the longest shared run of low bits between any pair is 11 bits (2170280596 / 2404401812), 1 pair of 435. Uniform low bits give 0.21 such pairs [assumed reference].
- **Stream overlap:** no row's 48-bit generator state reaches another row's within 2^20 steps, so no two rows walk the same stream.
- **Density around members:** forward and reverse survivors within ±2^16 of each row number 17.2 and 23.5 (E) and 16.9 and 25.0 (F), against slice averages of 16.8 and 23.8 (E) and 16.8 and 23.7 (F). At ±2^20 they are also at the average. Survivors do not concentrate near the 30.

## 4. What was not done

- **No GPU and no rig.** Everything here is CPU on VM101.
- **Recur over the whole record** (other slices, other sessions, other window sizes) as a rig run: not run. The CPU controls already cover 100 other anchors on the midday stream.
- The evening session was not used.
- **Peak memory** of the CPU runs was under 400 MB.

## 5. Scope, not built: a shuffled-draw rig control

**What it would measure.** The rig path's two-way count on a window with the run's values in another order, or on random values. CPU predicts it from §2.2 (ratio about 3.3–3.9 at F≥6 R≥5). Its added information is a fifth rig-to-CPU agreement on a non-real window. CPU already matches the rig on four runs.

**Dataset authority rules it must respect:**
- **A new file, never a published version.** `resolve_dataset_path` (`miner/dataset_authority.py:477-583`) accepts any non-alias, non-version-named file as case 3: "explicit", `version_id=None`, frozen on its own path and digest. It must not be named in the `daily3-<ts>-<digest12>.json` grammar (case 2 checks the digest against the name, `:543-551`). It must never touch `daily3.json` or `daily3_current.json` (module rule, `:37-42`).
- **Session-separated.** Midday and evening are separate streams and combined order carries no generator meaning (`BACKLOG.md` §5). Shuffle only within the midday series, and record which records moved.
- **Its own run root and `prng_analysis.db`.** The coverage ledger stores `dataset_sha256` as provenance, not as a partition key (`utils/seed_coverage_ledger.py:884-889`). A shuffled-dataset run in a shared database would certify the slice as covered, so it must run in a separate tree, as runs A–H do, with its own staging directory.
- **Fleet provisioning.** A miner-backed run verifies the dataset on every rig before dispatch (`run_start_dataset_gate`, `:1235` onward). The provisioning manifest must name the control file. [UNVERIFIED: whether the Stage 4 launcher path accepts a case-3 dataset end to end was not traced.]
- **Not a certifying run, and labelled as one:** its NPZ must never become any Step 2 input.

**What a rig cannot do without a code change:** the cross control (forward and reverse on different windows). Both phases read one residue window.

**Cost:** about 100 s per run on the rigs (runs E–H took 98–101 s). It needs the owner's go.

## 6. What each outcome predicts for pool hit rate against `k/1000` on unseen draws

**The outcome measured: the overlap is sieve arithmetic on shared values.**
- It predicts **nothing about pool hit rate.**
- It says the bidirectional count is 1.8–3.9× larger than an independence model gives, for any window, real or not. It does not say whether those rows carry signal.
- Because the same excess appears for shuffled and random windows, **the excess itself cannot be read as evidence for or against the draws containing structure.**
- **Population size is load, not quality** (`PITFALL_STATE_RECOVERY_YARDSTICK.md` §1, §5).

**The outcome not measured, had the excess depended on the draws** (larger for the real window than for shuffled or random ones):
- It would have predicted that rows passing on real windows carry draw-specific structure.
- So pools ranked from them would beat matched pools from control-window populations on unseen draws.
- The controls did not show that difference at the thresholds tested.
- **This is not a result about hit rate either way.**

**The only settling measurement:**
- Build pools from a real-window population (for example run H's 12,156 rows) through Steps 2–6.
- Score them on draws the sieve never saw, against `k/1000`.
- Optionally score matched pools from a shuffled-draw population alongside.
- **Only pools scored on unseen draws settle value.**

**Unchanged:** the main line of work, a baseline that hands Step 2 a population (run H at 0.30/0.30, owner 14:42). This lead does not gate it.

## Files

All files are in `/home/michael/tfm_worktrees/pwc-offset/logs/lead_bidir_20261122/` (the project tree, not `/tmp`):

| File | sha256 |
|---|---|
| `hyb.py` | `1ed11385319fafd3…` |
| `validate.py` | `0c45e022c27eb2b0…` |
| `reproduce_EF.py` | `0f3bc2ef94647efa…` |
| `controls.py` | `ce8a9b8323a05782…` |
| `neighbourhood.py` | `d8a31b9bcb7409be…` |
| `analyse.py` | `01fa2fe981213e46…` |
| `reproduce_EF.json` | `316630e3ae5adaa4…` |
| `controls.json` | `de0a38e7cab3a357…` |
| `analysis.json` | `f28543bb42769857…` |
| `neighbourhood.json` | `4840f356de2f411c…` |

- **Logs:** `validate.log`, `reproduce_EF.log`, `controls.log`, `analysis_controls.log`, `neighbourhood.log`, `density.log`.
- **Survivor sets** the rig run did not keep: `fwd_survivors_{E,F}.npy`, `rev_survivors_{E,F}.npy`.
