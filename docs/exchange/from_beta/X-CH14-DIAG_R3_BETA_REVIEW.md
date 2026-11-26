# X-CH14-DIAG · R3 · Team Beta before-commit review

## Prior record

I searched `docs/` and `docs/exchange/` with `rg` for Chapter 14 diagnostics, `grading_rules`, `expiry_mode`, `skip_active`, `catboost_info`, writes outside `--out`, `best_trial.params`, and failed compare children and their archives. The searches included the catalog, rulings, proposals, chapter audits, consolidated TODO, backlog and leads register. I read the named Chapter 14 task and owner records in `logs/e2e_session/`, the R1 and R2 exchanges, the LL6 review and closing note, the learning loop in `CLAUDE.md:43–66`, and the Chapter 13 owner amendment. I compared the 18 submitted paths between the detached base and main with `git diff`, and read the older Step 5 behavior with `git show 7f4505a:meta_prediction_optimizer_anti_overfit.py`.

**Finding 1. The governing record is incomplete.** `logs/e2e_session/TASK_CH14_DIAG_LAND_PREP_20261125.txt:2–6` appears as a search hit, but the request omits its operative requirements from the governing list. It requires confinement to `--out`, a pinned baseline and a mutant for each gate. The later CatBoost-specific task adds requirements and leaves that wider boundary in force. R3 also misses `docs/exchange/from_beta/X-E2E-GRADE_R1_BETA_REVIEW.md:55`, which already records that the stock compare route archives canonical artifacts after a failed child and can select by their score. That record governs how to interpret the observed CatBoost archive and the winner-restore check in Q1.

| Item | Label | Record |
|---|---|---|
| Evidence from R2, resubmitted with searches, and the R2 intake decision | ALREADY RECORDED | This thread's R1 and R2 reviews and R2 reply |
| Owner authority on defaults, benching, why-reports and housekeeping | ALREADY RECORDED | `OWNER_RULING_defaults_20261123.txt:4–10`, `OWNER_RULINGS_pending_list_20261124.txt:20–33` in `logs/e2e_session/` |
| Confinement, CatBoost write-through, and the broader Git-status blind spot | ALREADY RECORDED | Landing-prep task, R1 Q4, R3 disclosure 1, and `X-LL6-BUNDLE_R1_BETA_REVIEW.md:59` |
| Failed-child archive provenance | ALREADY RECORDED | `X-E2E-GRADE_R1_BETA_REVIEW.md:55` and R3 Q1 evidence |
| Best-trial rule and the invalid-expiry behavior | ALREADY RECORDED | `docs/CHAPTER_6_AUDIT.md:197–201`, this thread's R1 Q5, and `logs/e2e_session/LEAD_CODE_REVIEW_ch14diag_20261125.txt:3–5` |
| Independent checks of the sibling overwrite, temporary cache, invalid expiry and compare archives | NEW | Q1, Q5 and Q7 below |
| CHANGES A RULING | None | Michael retains the landing decision and the authority to flip policy defaults. |

Search exclusions were the rigs, ser8 archives, remote-only state, live run ledgers, and other agents' worktrees. Host checks were limited to the named records, the submitted worktree and the local CPU environment, with temporary fixtures under `/tmp`.

## Review record

**Date** 2026-11-25 PST. **Main read** `1bb75f0d400c5291af361f84b809d3dddc02afaf`. **Package read** `/home/michael/tfm_worktrees/ch14diag-land2`, detached at `579dbb7e558da1d90032967ca7b3e28054180e7a`. All 18 hashes matched the submitted manifest and READY hash. The 18 paths have no committed differences between those two bases. The behavioral baseline used by the suite is `7f4505aaccebdd284393eb3565e75d09d1c0f3b4`.

Main `git status --porcelain` at intake

```text
 M docs/exchange/LEDGER.md
?? docs/exchange/from_beta/X-CH14-DIAG_R2_BETA_REVIEW.md
?? docs/exchange/to_beta/X-CH14-DIAG_R2_ALPHA_REPLY.md
?? docs/exchange/to_beta/X-CH14-DIAG_R3_ALPHA_REPLY.md
```

Reviewed worktree `git status --porcelain --untracked-files=normal`

```text
 M agents/contexts/advisor_bundle.py
 M diagnostics_llm_analyzer.py
 M meta_prediction_optimizer_anti_overfit.py
 M parameter_advisor.py
 M train_single_trial.py
 M training_diagnostics.py
 M training_health_check.py
?? data/
?? docs/drafts/ch14_v1/
?? docs/drafts/loop_v1/BETA_R2_CH14_DIAG_SUMMARY_20261125.md
?? docs/drafts/loop_v1/DRAFT_WATCHER_CD07_20261123.diff
?? docs/drafts/loop_v1/READY_LAND_CH14_DIAG_20261125.md
?? docs/drafts/loop_v1/READY_LOOP_DIAG_20261123.md
?? docs/drafts/loop_v1/READY_LOOP_DIAG_2_20261123.md
?? docs/drafts/loop_v1/X-CH14-DIAG_R2_ALPHA_REPLY.md
?? docs/drafts/loop_v1/X-CH14-DIAG_R3_ALPHA_REPLY.md
?? tests/ch14_cpu_patch.py
?? tests/ch14_wave1_mutants.py
?? tests/ch14_wave2_mutants.py
?? tests/test_ch14_diag_land_defaults.py
?? tests/test_ch14_diag_wave1.py
?? tests/test_ch14_diag_wave2.py
```

## Positions

| Question | Position | Severity |
|---|---|---|
| Q1 · WATCHER's Step 5 route at defaults on CPU | NO OBJECTION | — |
| Q2 · Best-trial and worker-boundary gates | NO OBJECTION | — |
| Q3 · `diagnostics_options=None` | NO OBJECTION | — |
| Q4 · Corrected health wording | NO OBJECTION | — |
| Q5 · Test isolation and V1–V4 | OBJECTION | LIVE — The single-part suite writes outside `--out` in its own control and the isolation gates can still report success after an overwrite in a sibling directory. |
| Q6 · Real NN health outcome | NO OBJECTION | — |
| Q7 · Default skip behavior and invalid expiry | NO OBJECTION | — |

## Reasons

### Q1 · WATCHER's Step 5 route on CPU

The compare and single parts close R1 Q1 for default parity on the supplied CPU fixture. The compare part exercises the manifest's `--compare-models --enable-diagnostics` route with four recorded child calls, stable output comparisons, the run's own winner restore, and the context-free health call on identical input reports. The saved 26/26 includes S07 and S08 reaching and failing their target checks. The single part supplies successful per-family CPU runs. Compare-file parity is by content where the baseline repeat agrees and by structure for the selected variable artifacts. `agent_manifests/reinforcement.json:173–184`, `agents/watcher_agent.py:3182–3190`, `tests/test_ch14_diag_land_defaults.py:232–250,677–777,1023–1030`, `data/ch14_land2/q4/logs/defaults_compare.log:13–33`.

In all five compare archives I checked, CatBoost exited `1` and four of its archived `best_model*` files were byte-identical to the preceding XGBoost archive. The baseline CLI log identifies the absent CUDA device. That reproduces the existing archive behavior in Finding 1. The restore check establishes a copy from the recorded winner's archive. Eligibility and provenance require a successful child and a fresh matching checkpoint and sidecar. A successful CatBoost compare child and exact unseeded NN value parity remain **UNAVAILABLE**. `meta_prediction_optimizer_anti_overfit.py:232–242,274–334`, `7f4505a:meta_prediction_optimizer_anti_overfit.py:268–309`, `data/ch14_land2/q4/defaults_compare/compare/base/cli.log:365–391`.

### Q2 · Best trial and worker boundary

Both R1 gaps close on the exercised path. The new test compares the final NN argv with the best parameters from a copy of that run's study. The worker test then calls real `main()` with one fixed final argv and NPZ across both versions, and compares decoded arrays, trainer arguments, saved weights and predictions. The recorded S05 run passes the old membership check and fails the study check. S06 passes the old structure and argv checks, while the new check observes a learning rate of `0.0005628083782686422` versus `0.00037520558551242813`. `meta_prediction_optimizer_anti_overfit.py:1850–1859,2610–2628`, `train_single_trial.py:978–990,1033–1042`, `tests/test_ch14_diag_land_defaults.py:392–468,604–674`, `data/ch14_land2/q4/logs/defaults_single.log:89–94`.

Variations can still pass. An S05 variant that changes fold scoring so a different trial legitimately leads the recorded study can pass the best-trial check. An S06 variant confined to fold workers can pass the seeded final-call check, as can a change confined to the value of an unexercised option or to GPU execution. The current test uses the baseline's final argv and a substituted NPZ. These are limits of the fixture and preserve the two scoped conclusions. `meta_prediction_optimizer_anti_overfit.py:2150–2166`, `tests/test_ch14_diag_land_defaults.py:472–503,611–636`.

### Q3 · Explicit `None` at the trainer boundary

The rule is sound for this named keyword. Binding `diagnostics_options=None` produces the same parameter value as omission because the signature defaults to `None`. The normalizer removes only that explicit null and preserves a populated options record. Legacy reaches the trainer with `_diag_opts = None`. `train_single_trial.py:493–496,1008–1041`, `tests/test_ch14_diag_land_defaults.py:435–443,594–601`.

### Q4 · Corrected health wording

Accurate. The health implementation validates settings before it branches on grading rules. Its public wrapper catches a bad policy value and returns `absent` / `PROCEED_WITH_NOTE` with a module-error note. WATCHER's post-success health handler logs the note and allows progress. This concerns the health call after Step 5. The compare dispatcher has its own earlier policy read and failure path, covered in Q7. `training_health_check.py:130–178,223–250,952–958,980–991`, `agents/watcher_agent.py:2570–2586,3182–3190`.

### Q5 · CatBoost fix and the remaining isolation objection

The specific R1 CatBoost write-through is closed for the normal defaults run tree. `build_tree` excludes the output roots and links tracked input entries. The saved old-policy control writes through to the CatBoost fixture and the snapshot detects it. The new-policy control leaves that fixture unchanged. V1 now checks the LL-3 marker from inside the shared fixture, and the saved stale-pin run fails at the credited assertion. The recorded V2–V4 controls exercise exact child counts, a report delta under `diagnostics_outputs/`, and the deterministic grade/action assertion. `tests/test_ch14_diag_land_defaults.py:93–148,157–201,786–865,868–932,1003–1022,1086–1092`, `tests/test_ch14_diag_wave1.py:64–100`, `data/ch14_land2/q4/logs/v1_pin_41ad509.log:22–46`, `data/ch14_land2/q4/logs/defaults_single.log:95–102`.

The wider isolation objection is LIVE. In a temporary Git repository I overwrote an existing file in a sibling of the selected `--out` under `data/` and ran the submitted snapshot and finishing functions. Both isolation gates passed. The snapshot prunes `data/`, and the status check accepts new ignored `data/` entries and compares status lines only. In a separate scratch check, the installed LightGBM import created a Matplotlib cache under the inherited `TMPDIR` outside the chosen output when the configured cache was unwritable. It cleaned the directory on exit and Git status stayed identical. The Step 5 LightGBM path imports that same library and the harness inherits the cache environment. The supplied runs' logs establish the CatBoost snapshot outcome for the watched paths. A sibling overwrite during those training runs is **UNAVAILABLE**. `tests/test_ch14_diag_land_defaults.py:157–183,232–250,1143–1155`, `meta_prediction_optimizer_anti_overfit.py:1063–1065`.

The single-part runner also creates its negative-control fixture at the worktree root after the general snapshots and checks. This is an intentional write outside `--out` in a normal invocation. Run the old-policy control against a disposable source root inside `--out`. Route temp and library caches under `--out` before imports and subprocesses, and snapshot sibling paths with only the selected output subtree excluded. Require a control that overwrites an existing sibling file and fails the boundary check. The LL6 closing note records the relevant pattern. `tests/test_ch14_diag_land_defaults.py:868–894,916–932,1137–1158`, `scripts/diagnostics/e2e_walk_forward.py:383–395`, `docs/exchange/to_beta/X-LL6-BUNDLE_R2_ALPHA_CLOSING_NOTE.md:55–63`.

### Q6 · Real NN health result

Keep the real report's grade/action pair as an observation. On the same report, retain the existing base-versus-new legacy identity gate. The deterministic fixture establishes `critical/RETRY` → `ok/PROCEED` for the corrected best-round case, and V4 fails that assertion. The final NN training report produced `critical/RETRY` under both rule sets. Qualify the READY plain-summary claim at line 241 as the synthetic fixture result. If the suite is also meant to assert that fixed grading successfully processed every real NN report, require a valid fixed result without a module-error outcome and allow either grade. `tests/test_ch14_diag_land_defaults.py:1023–1041,1063–1092`, `training_health_check.py:223–239,595–627`, `data/ch14_land2/q4/logs/defaults_single.log:97`, `docs/drafts/loop_v1/READY_LAND_CH14_DIAG_20261125.md:234–244`.

### Q7 · Skip expiry and the fallback

CD-09 changes compare family selection independently of the diagnostics level. With `expiry_mode` omitted, `until_reset` uses the same threshold predicate as the baseline for the current numeric counts. The checked policy sets the threshold to three and omits `expiry_mode`. The NN registry fixture has count one. I verified the old and new predicate agree at counts zero through four. In scratch, `expiry_mode: days` raised from both the compare pre-dispatch code and `is_model_skipped`. The context-free health entry point returned the module-error result described in Q4. The compare branch also catches JSON and I/O errors around registry handling, and `ValueError` from the policy is outside its catches. `watcher_policies.json:165–169`, `diagnostics_outputs/model_skip_registry.json:1–9`, `7f4505a:meta_prediction_optimizer_anti_overfit.py:156–184`, `meta_prediction_optimizer_anti_overfit.py:154–190,3254–3280,3398–3399`, `training_health_check.py:112–113,901–918,1232–1256`.

Keep the explicit error for an invalid supplied expiry mode. For WATCHER, the compare CLI's nonzero exit is a Step 5 failure, which WATCHER handles and escalates. Its current training-health integration calls `check_training_health()` after a successful step. It does not use `is_model_skipped()` there. A fallback would be a separate policy decision because the old rule could keep a family benched indefinitely after an operator attempted to configure an expiry. `agents/watcher_agent.py:101–109,1100–1128,2463–2477,3182–3190`.

## Checks and limits

I verified the package hashes and base diff, inspected the submitted code and saved single/compare results, compared the five saved CatBoost and XGBoost archives, and ran the scratch isolation, dependency-import and expiry probes in the torch environment. The full Python and mutation suites were read evidence. An independent full suite replay, an end-to-end WATCHER run, real-data parity and GPU evidence are **UNAVAILABLE**. This review measures dispatch, training and grading behavior on fixtures. Held-out prediction-pool hit rate against `k/1000` remains unmeasured here.

R1 Finding 1 is recorded in the updated READY. The November 24 owner ruling assigns the housekeeping decisions to the lead and gives their direction. Activation of those settings and the separate benching switch remains tied to their landing. `logs/e2e_session/OWNER_RULINGS_pending_list_20261124.txt:20–33`, `docs/drafts/loop_v1/READY_LAND_CH14_DIAG_20261125.md:102–122`.
