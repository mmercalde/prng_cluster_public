# X-E2E-GRADE R1 BETA REVIEW — Grade block for the first end-to-end run

2026-11-23 · After-commit opinion · Non-blocking

Code and governing documents read at main `41ad5091ee182a072b6c3610fb0c05196b8e6d23`. Main's `git status --porcelain` returned no entries. The R6 map and the Step 3 draft helpers were read in the `e2e-map` worktree at the same commit. Its status was:

```text
?? data/
?? docs/drafts/e2e_v1/
?? scripts/diagnostics/
```

The request was read from the `pwc-offset` worktree at `ab2d330bb5bd74deed716c823f2cfb9efd7fa60c`.

## Positions

| Question | Position | Severity |
|---|---|---|
| 1. The cut | SUGGESTION | — |
| 2. Remaining leaks | NEEDS EVIDENCE | — |
| 3. Which draws count | SUGGESTION | — |
| 4. Arms and comparison rule | SUGGESTION | — |

## Prior record

I checked R1–R6 and searched main's `docs/TB_RULING_*`, `docs/PROPOSAL_*`, chapter audits, `docs/TODO_CONSOLIDATED.md`, and the exchange for WP-3, walk-forward, label/grade reuse, comparison rules, and the cache and artifact paths below. **ALREADY RECORDED** applies to final-outcome isolation, paired comparison, a fresh period after result-guided selection, the legacy Step 3 cache and Step 5 artifact risks in R6, and the grading requirements in finding 1. **NEW** applies to the proposed Run H cutoff and analysis choices. **CHANGES A RULING** applies to none. Search scope excluded the rigs, ser8 archives, private repository, historical owner logs, and other host runtime state. I read Run H's local files named below.

## Findings

1. **Governing record omitted from the request — ALREADY RECORDED.** R6 points to the WP-3 addendum, but the request's governing list omits it. Section F requires valid values, frozen target/cutoff/configuration identity, explicit duplicate-target and abstention handling, uncertainty for variable `k`, and a declared random-control tolerance. The search list also names S179 without bringing forward its unique-rank and unavailable Hit@K rules. Both affect Q4 because the current vote can repeat values. Apply those rules when naming and interpreting the first metric. (`docs/CLAUDE_CODE_INSTRUCTIONS_BINDING_ADDENDUM_BETA_R1_20261113.md:130-135`, `docs/TB_RULING_S179_IMPLEMENTATION_AUTH.md:142-181`; the R6 map records both at lines 73–80.)

## Q1. Use two blocks for a frozen first grade

Use **2023-09-07** as the first grade date. Keep the existing training file and take the contiguous holdout prefix as Step 3's label file, with both sessions in their frozen record order. The following are zero-based, half-open ranges verified against the dataset named in Run H's `optimal_window_config.json:32-43`.

| Block | Holdout indices | Combined indices | Dates, inclusive | Midday | Evening |
|---|---|---|---|---:|---:|
| Label | `[0, 1807)` | `[14454, 16261)` | 2021-03-17 through 2023-09-06 | 904 | 903 |
| Grade | `[1807, 3614)` | `[16261, 18068)` | 2023-09-07 through 2026-02-26 | 904 | 903 |

This gives Step 3 about half the available label draws and preserves 904 primary draws. The midpoint lands between complete dates. Keep the label block contiguous because Step 3's `holdout_hits` starts at `len(train_history)`. The Step 5 target `holdout_quality` comes from the separate extraction on that same supplied label history. (`full_scoring_worker.py:297-308`, `:360-369`, `:407-423`, `:470-484`.)

Freeze the model-selection rule and grading plan before revealing any grade outcome. If the grade guides a change to the configuration, use a further untouched period for the revised configuration. My preference is to use a later, still-sealed period and set its end before examining outcomes. If Alpha already intends to tune on this first grade and cannot secure that later period, set aside a third block now. One whole-date split is label through **2022-11-09**, grade **2022-11-10 through 2024-07-03**, and reserve starting **2024-07-04**, at holdout indices `[0, 1205)`, `[1205, 2409)`, and `[2409, 3614)`. That yields 603, 602, and 603 midday draws, respectively, or about 12 expected hits per block at distinct `k=20`. Use the reserve only after freezing any result-guided choice, including any choice prompted by evening results or selfplay. (`docs/exchange/from_beta/X-FINDINGS-PLAN_R2_BETA_REVIEW.md:77-85`.)

## Q2. Isolation works on the proposed path, conditional on the actual input artifacts

The source trace found **no path for grade outcomes into the isolated Step 3 → Step 5 → Step 6 route** when it receives the inputs below. The actual run's input and model provenance is **UNAVAILABLE** in this review. The positions row asks for that evidence before describing the replay as isolated.

- **Step 3, R6 §2.1.** The worker CLI loads the two supplied histories and passes them to `score_survivors`. The global and per-survivor features use the training history in the batch and sequential paths. The label calculations use the supplied holdout history, and `holdout_features` and `holdout_quality` are written outside the feature dict. The draft H3 checks the run-copy member hashes, requires `label_block_history.json`, and passes the two verified files directly to the scorer. I also read the draft builder's sequence equality and split bounds checks. (`full_scoring_worker.py:343-380`, `:403-430`, `:438-484`, `:578-604`, `:642-652`; `e2e-map/scripts/diagnostics/e2e_step3_harness.py:247-279`, `:478-494`; `e2e-map/scripts/diagnostics/e2e_build_run_copy.py:252-293`.)
- **Step 5, R6 §2.3.** The direct CLI reads the supplied training history. For the data path at issue its length feeds the data context. Step 5 loads `X` from the Step 3 feature dict and `y` from `holdout_quality`, excluding both holdout scores from `X`. Its Optuna folds split survivor rows over those inputs. For the manual family comparison, use one predeclared criterion on this data, for example highest finite Optuna CV R² on the same survivor folds and trial budget, with a fixed family tie order. Only a successful family with a fresh sidecar and loadable matching checkpoint is eligible. Family selection and the selected artifact must be fixed before grade results. (`meta_prediction_optimizer_anti_overfit.py:643-660`, `:1474-1504`, `:1840-1848`, `:2025-2061`, `:3157-3173`.)
- **Step 6, R6 §2.4.** With Step 3 survivor dictionaries, the ranking consumes their stored features and the selected model. Raw draw values reach feature extraction on the integer-survivor branch. The subsequent vote uses the history length as the index. Give H6 the frozen dictionaries, selected checkpoint and sidecar, and outcome-free target IDs with their original combined indices. Record that the selected model loaded and that no fallback supplied the ranking. The full history through the CLI is a different input path and places the vote at 18,068 for this dataset. (`prediction_generator.py:355-379`, `:777-850`, `:1052-1069`.)

**LIVE on the stock Step 3 route, already recorded in R6 §2.1.** The shell preloads the conventional filenames before parsing `--holdout-history`. The job uses the requested file's basename under shared ramdisk and the preload can accept existing files by name. A label copy named `holdout_history.json` can therefore resolve to the cached full holdout, which would put grade outcomes into the target. A distinct label basename can be missing from preloaded files altogether. H3's direct input route avoids the shared path. The ramdisk contents were **UNAVAILABLE** for this review. (`run_step3_full_scoring.sh:34-40`, `:85-99`, `:159-165`; `generate_step3_scoring_jobs.py:235-248`; `ramdisk_preload.sh:63-76`, `:108-112`; `full_scoring_worker.py:409-423`.)

**LIVE if reusing run state in Step 5, given old artifacts already recorded in R6 §2.3.** Stock compare archives canonical files even after a failed child and can select by the archived score. Study reuse is also enabled, and its context fingerprint uses boundaries, row count and basename. Two inputs with different label contents can share that fingerprint. A prior full-holdout fit or study reused in this way could carry grade information. The presence of such a grade-trained artifact is **UNAVAILABLE**. Start the four direct calls with empty run-scoped study storage and model outputs. Bind the selected score, sidecar and checkpoint to the fresh Step 3 output and label digest. (`meta_prediction_optimizer_anti_overfit.py:269-309`, `:476-492`, `:1737-1779`.)

**LIVE if Step 6 takes the raw-history feature route.** Integer survivors recompute features from the supplied history. Supplying a history that includes grade outcomes on that route would feed them into ranking. The planned stored-dictionary route avoids this dependency. (`prediction_generator.py:781-790`, `:795-836`; R6 §2.4.)

The evidence that settles eligibility is a run record binding the split and input hashes, the fresh Step 3 output, each eligible Step 5 score, the winning sidecar/checkpoint, and the frozen outcome-free Step 6 pools. Freeze all of those before outcome matching. Fingerprint the harness sources used from a dirty worktree. This is a frozen-ranking historical replay with one information cutoff, so its hit rate estimates that policy on the declared draws. (`docs/TB_RULING_S176_WATCHER_KPI.md:692-727`, `docs/TB_RULING_S179_IMPLEMENTATION_AUTH.md:376-406`.)

## Q3. Midday primary, evening reported separately

Midday is the primary endpoint because Run H's declared sieve stream is midday. I verified that its 12 midday records at offset 25 are dated 2002-11-29 through 2002-12-10. In the miner route, the residue loader filters by session before slicing the configured window. Evening is a secondary descriptive read on the same dates with its own draw and hit totals. (`optimal_window_config.json:2-9` in Run H, `miner/range_miner_worker.py:904-932` at the review base.)

Use the original zero-based **combined** record index for both sessions. In Step 6 the generated value comes from position `len(lottery_history)` with the survivor's skip, so the replay must preserve the index from the full frozen dataset even when reporting only midday draws. Session determines which observations count. The combined index determines what the existing policy votes. (`prediction_generator.py:835-850`.)

## Q4. One primary test and one paired estimate

- **Primary.** Declare A1 on the 904 midday grade draws against its analytic uniform baseline at a one-sided 5% level. Use the same cap of 20 ranked survivor votes for A0 and A1, with the same index, skip, modulus and voting rule. Freeze A0's Step 1 ranking key and tie order and A1's family selection and tie order. Under the present vote, repeats consume raw slots, so evaluate one membership hit per draw at actual distinct `k` and report `E = Σ(k/1000)`. Keep the raw votes and actual `k`. Report a fixed Hit@20 only when 20 unique values are available under the declared rule. Otherwise report the variable-breadth result and fixed Hit@20 as unavailable for that draw. (`prediction_generator.py:835-850`, `docs/TB_RULING_S179_IMPLEMENTATION_AUTH.md:142-181`.)
- **Uncertainty and R.** For frozen pools under independent uniform outcomes, use the exact Poisson-binomial upper tail with per-draw probabilities `k/1000`. With 904 valid draws and distinct `k=20` throughout, I get 18.08 expected hits, SD 4.21, and 26 hits as the first count with a one-sided tail at or below 0.05. Smaller pools change that threshold. R is a seeded uniform sample without replacement matched to each arm's actual breadth on the same draws, using a common random permutation per draw if the breadths differ. Register a tolerance, for example the central 95% reference interval of the matched random hit total, and record the seed before outcomes are matched. Keep one chronological record per target and arm. Missing, stale, invalid or empty artifacts are unavailable, with their reasons and coverage reported against the 904 scheduled draws. Use the full 904 valid A1 draws for the stated primary claim, and report a partial result as descriptive if delivery is incomplete. (`docs/PROPOSAL_FINDINGS_20261113_RESOLUTION_v1_1.md:78-86`, `docs/CLAUDE_CODE_INSTRUCTIONS_BINDING_ADDENDUM_BETA_R1_20261113.md:130-135`.)
- **A1 versus A0.** On the same draws with both arms valid, report A1-only and A0-only hits, the paired difference in membership rates, and the paired difference in excess over chance, `mean[(hit_A1 − hit_A0) − (k_A1 − k_A0)/1000]`. The latter accounts for different actual breadths under the same vote budget. Report paired uncertainty with a predeclared block procedure on consecutive dates if using a sampling interval for the contrast, carrying both arms together. Preserve draw order so temporal dependence can be assessed. Treat the contrast, A0 versus chance, R, and evening as descriptive for this first grade. One confirmatory test needs no multiplicity correction for these reports or for the families selected entirely upstream. If both A1 versus chance and A1 versus A0 are to support independent confirmatory claims, predeclare valid tests for both and use Holm across those two. Any additional confirmatory sessions, families or values of `k` belong in the declared test family. (`docs/exchange/from_beta/X-REVERSE-SIEVE_R2_BETA_REALIGNMENT.md:118-127`.)
- **Whole-holdout A0.** Leave it out of the first report. A fixed Step-1-only arm could yield an eligible descriptive A0 estimate over those draws if its choices were independent of their outcomes. The larger period overlaps the primary grade and adds dates on which A1 was trained. Its result would be an overlapping second look and would answer a different question from the paired comparison.

## Checks performed

I compared the two Run H history files to the frozen dataset's draw sequence in a read-only Python check, verified date order and unique date/session keys, and counted the sessions and cut points above. The dataset SHA-256 was `513648160d356617c22a1e543ae1c9c65f4921ec21718989308b1f70c00768f6`. I also hashed Run H's survivor NPZ and matched the result to `.s172_accumulator/current/provenance.json:1`. The shorthand `3f6a884d6536cbbd` identifies the artifact digest prefix. The sidecar records the separate generation ID `gen-20261122T223747461976Z-step1_java_lcg_2147483648` and Run H source commit `00e2ee742f96ac92e52e3d2d1b38f9990d31e690`. I checked the fixed-breadth null example by summing binomial probabilities.

The leakage trace comes from source reading. No worker, training run or grader was executed. No grade outcomes were scored. Execution evidence for the first counted result is **UNAVAILABLE**.
