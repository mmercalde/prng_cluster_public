# X-CH14-DIAG · R1 · Team Beta before-commit review

## Prior record

I searched `docs/` and `docs/exchange/` with `rg -n -i` for `CH14.DIAG|diagnostics.level|grading_rules|skip.registry|benching|why.report|stale.report|history.prune|catboost_info`, including the rulings, proposals, audits, backlog, TODO and leads register. I searched the named `logs/e2e_session/` task and owner records for Chapter 14, model benching and defaults. I compared `41ad509`, `1ba3d6f`, `7f4505a` and `579dbb7` with `git diff` and `git show`.

**Finding 1. The prior record missed the November 24 owner ruling on model benching and housekeeping.** `logs/e2e_session/OWNER_RULINGS_pending_list_20261124.txt:20–33` says poor health should trigger a why-report, retains benching behind an off switch when that work lands, defers the automatic why-report trigger and queue, and gives the lead the archive move, dead-unit and log decisions at landing. `CLAUDE.md:49–57` restates the model-family rule. The READY note recognizes the rule (`READY_LAND_CH14_DIAG_20261125.md:294–297`). The lead's landing-prep instructions also explicitly require the full `7f4505a` baseline, confinement to `--out` and mutation coverage (`logs/e2e_session/TASK_CH14_DIAG_LAND_PREP_20261125.txt:2–6`). This package still applies the existing benching threshold (`meta_prediction_optimizer_anti_overfit.py:148–188`, `training_health_check.py:1028–1076`). The owner ruling says that change flips when landed. Treat the retained skip behavior as an existing status and keep implementation of the off switch tied to the owner decision for that work.

| Item | Label | Record |
|---|---|---|
| CD-00 through CD-13, CD-23, report and grading switches, and the earlier diagnostics defects | ALREADY RECORDED | Chapter 14 plan and wave READY notes, `docs/CHAPTER_14_AUDIT.md` and `docs/CHAPTER_CORRECTIONS_REPORT_20261111.md:344–401` |
| Default flips and the separate owner decision on model benching | ALREADY RECORDED | `logs/e2e_session/OWNER_RULING_defaults_20261123.txt:4–10`, `logs/e2e_session/OWNER_RULINGS_pending_list_20261124.txt:20–33`, `CLAUDE.md:43–66` |
| NN run-to-run variability and the earlier `models/reinforcement` link incident | ALREADY RECORDED | `READY_LAND_CH14_DIAG_20261125.md:182–264` |
| CatBoost prediction comparison, remaining write through a `catboost_info` link, and gate limits below | NEW | Q1, Q2 and Q4 |
| CHANGES A RULING | None | The owner decides when each policy change lands. |

Search exclusions were the rigs, ser8 archives, the private remote and host paths beyond the named trees, evidence and CPU environment. The owner intent document named at `CLAUDE.md:70–73` was unavailable at that path. I read the learning-loop section in `CLAUDE.md` and the owner ruling on disk.

## Review record

**Date** 2026-11-25 PST. **Main read** `579dbb7e558da1d90032967ca7b3e28054180e7a`. **Code reviewed** the prepared files in `/home/michael/tfm_worktrees/ch14diag-land` at detached base `7f4505aaccebdd284393eb3565e75d09d1c0f3b4`. All 17 listed hashes and the separate READY note hash matched. All 18 paths were unchanged between the two base revisions. Rechecked before writing.

Main `git status --porcelain` at intake

```text
 M docs/exchange/LEDGER.md
?? docs/exchange/from_beta/X-LL6-BUNDLE_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-REGISTRY-REPAIR_R1_BETA_REVIEW.md
?? docs/exchange/to_beta/X-CH14-DIAG_R1_ALPHA_REVIEW_REQUEST_ch14_diag_landing.md
?? docs/exchange/to_beta/X-LL6-BUNDLE_R1_ALPHA_REVIEW_REQUEST_inference_bundle.md
?? docs/exchange/to_beta/X-REGISTRY-REPAIR_R1_ALPHA_REVIEW_REQUEST_registry_parts2_3.md
```

Reviewed worktree `git status --porcelain` at intake

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
?? docs/drafts/loop_v1/DRAFT_WATCHER_CD07_20261123.diff
?? docs/drafts/loop_v1/READY_LAND_CH14_DIAG_20261125.md
?? docs/drafts/loop_v1/READY_LOOP_DIAG_20261123.md
?? docs/drafts/loop_v1/READY_LOOP_DIAG_2_20261123.md
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
| Q1 · Default unchanged | NEEDS EVIDENCE | — |
| Q2 · NN gate | NEEDS EVIDENCE | — |
| Q3 · Merge | NO OBJECTION | — |
| Q4 · Gates and isolation | OBJECTION | LIVE — on this checkout the defaults harness can overwrite existing CatBoost logs outside `--out` and still report that isolation passed. |
| Q5 · Dark landing | NO OBJECTION | — |

## Reasons

### Q1 · Default unchanged

I found no default training regression in the routes I checked. WATCHER defaults to comparison with diagnostics enabled and persistent workers off (`agent_manifests/reinforcement.json:173–184`). It calls the health check without run context (`agents/watcher_agent.py:3175–3190`). At that configuration the new report options resolve to `None` and child flags to `[]` (`training_diagnostics.py:142–171`). The final tree reporter returns at legacy (`meta_prediction_optimizer_anti_overfit.py:2777–2792`). In a scratch comparison with child training stubbed, I got identical family selection, normalized child commands, S88 environment and compare-summary bytes for the base, omitted level and explicit legacy. The current registry count of one dispatched four families. A fixture at the configured threshold of three dispatched the three tree families on all sides (`diagnostics_outputs/model_skip_registry.json:1–9`, `watcher_policies.json:165–169`, `meta_prediction_optimizer_anti_overfit.py:148–188,220–299`). The seven wave-1 health fixtures passed against the base, including their registry counts and archive payloads (`tests/test_ch14_diag_wave1.py:140–181`).

Alpha's final evidence covers the single-family CLI on synthetic CPU inputs. It exercises the health readers against the same base reports, with a separate synthetic NN report, and leaves the real WATCHER compare-models training and its artifacts unverified (`tests/test_ch14_diag_land_defaults.py:139–157,395–449,476–502`). I read the five saved CatBoost models and ran inference on the full 240-row fixture in the sidecar's 89-feature order. Base repeat, omitted level, explicit legacy and full each gave identical predictions to base, with a maximum absolute difference of `0.0`. That closes the prediction question on this fixture for the differing CatBoost model files. Exact NN value parity and an end-to-end comparison with current WATCHER dispatch remain **UNAVAILABLE**. The evidence supports sampled parity, including the compare routing and legacy grading checks, and falls short of the exact claim in Q1.

### Q2 · NN gate

The tests establish normalized command and JSON key-path parity for the exercised CPU runs. The health comparison uses the base run's report, so it tests the reader on identical input. The seeded trainer test calls `train_neural_net()` directly, bypassing the worker's `main()` (`tests/test_ch14_diag_land_defaults.py:197–233,237–295,425–449`, `tests/test_ch14_diag_wave1.py:545–568`).

A realistic CLI-only regression is a different learning rate applied after `json.loads(args.params)` in the worker and before the call to `train_neural_net`. The worker sees the same argv and emits the same kinds of artifacts, while the optimizer consumes the changed rate and can produce different weights and trial scores. That path bypasses the seeded tests and can pass the structure checks (`train_single_trial.py:978–990,1033–1042,596–607`). I also replaced the recorded final call's three-layer winner with the other four-layer proposal in a copy of the argv trace. The gate accepted it. It checks membership in the proposals, while Step 5 is meant to pass `study.best_trial.params` (`tests/test_ch14_diag_land_defaults.py:266–295`, `meta_prediction_optimizer_anti_overfit.py:1849–1873`). Both are coverage gaps. I found no such learning-rate or winner-selection regression in the reviewed code.

To establish the stronger claim, compare the four decoded NPZ arrays and arguments at the worker-to-trainer boundary for base and new, then compare tensors and predictions with a test-only seed in the real worker CLI. Check the final child's parameters against the best trial from its own study. These checks can preserve production RNG behavior.

### Q3 · Merge

I independently ran `git merge-file -p` on the optimizer blobs from `7f4505a`, `41ad509` and `1ba3d6f`. The clean result matched the submitted optimizer byte for byte. The LL-3 arguments, resolver and constructor routing are present. The diagnostics parser builds its options alongside them (`meta_prediction_optimizer_anti_overfit.py:1502–1527,3182–3251,3312–3349`). The NN final worker keeps its existing arguments and receives the extra diagnostics flags only at full (`:2610–2645`). The two inline calls occur after their tree fits and use a full-only, best-effort writer (`:1868–1883,2777–2819,2854–2864`). The baseline already bypasses the LL-3 constructor when routing a compare parent, and the new compare call still omits those opt-in LL-3 fields (`7f4505a:meta_prediction_optimizer_anti_overfit.py:3153–3173`, reviewed `:3254–3280`). That has no effect with the current manifest defaults.

### Q4 · Gates and isolation

- **Verified mutation evidence.** The stored logs for M22–M24 and W30–W36 each contain the credited assertion failure. The summaries record positive marker hits for all ten (`tests/ch14_wave1_mutants.py:125–136,165–184`, `tests/ch14_wave2_mutants.py:146–173`). W34 now detects the sibling archive in its fixture. W35 reaches its tagged fallback assertion (`tests/test_ch14_diag_wave2.py:400–413,506–513`). S01 adds reports, S02 changes the LightGBM model, sidecar and study, S03 changes the fixed fixture's grade, and S04 changes the final NN child command. All reached their marker and the three Step 5 mutants exited zero in the recorded runs (`tests/test_ch14_diag_land_defaults.py:456–474,504–545`, `data/ch14_land/defaults_final/summary.json` and the mutant logs). Those are credited kills for the tested cases.
- **Pin and vacuity.** I reproduced the two credited precondition failures on the short SHA and the full branch SHA. The same precondition passes with the full `41ad509` SHA, which lacks the named markers and is the stale base the lead required replacing. The shipped literal is correct, but the wave-1 precondition only checks a full SHA and missing markers. The identity tests use the shared fixture without requiring the separate precondition test to run (`tests/test_ch14_diag_wave1.py:35–39,64–88,173–181`). The landing gate accepts two empty child traces, skips the full-argv check in that case, and treats a difference anywhere in the artifact set as a report change (`tests/test_ch14_diag_land_defaults.py:237–273,425–450`). The recorded NN traces have five calls, and the recorded tree health outcomes are all `absent/PROCEED`. The tree `fixed_changes_health` rows pass because the result dictionaries differ, even though grade and action agree. Require the expected NN call count, a genuine report delta, and the specific expected severity/action in cases that assert a grading change.
- **LIVE isolation failure.** `build_tree` symlinks a pre-existing top-level `catboost_info/` into every run because its exclusion list omits that directory. The main checkout already has it and `.gitignore:132` ignores it. The optimizer fits CatBoost without overriding the training log directory (`tests/test_ch14_diag_land_defaults.py:57–58,82–99,147–157`, `meta_prediction_optimizer_anti_overfit.py:1125–1140`). In a disposable repository under `/tmp`, the actual `build_tree` plus a two-iteration CPU CatBoost fit overwrote a pre-existing `catboost_info/time_left.tsv` outside `--out`. The only new status line was `!! data/`, and the suite's final predicate passed (`tests/test_ch14_diag_land_defaults.py:373–375,547–552`). This is a second writable link, separate from the recorded `models/reinforcement` incident. An overwrite of an already-ignored or already-modified file can preserve the same status line. Keep writable paths local and regression-test this case with a pre-existing log and content checks outside the output boundary.
- **Other worktree reads.** The Chapter 14 manifest check reads from its own worktree, and the baseline fixtures use that worktree's Git objects (`tests/test_ch14_diag_wave2.py:37–41,85–94,670–677`, `tests/test_ch14_diag_wave1.py:64–75`). I found no hardcoded other-worktree reads in the Chapter 14 suites under the recorded invocations. The two optional `--runh-root` gates in the separate LL suite require a caller-supplied evidence path (`tests/test_ll_feature_settings.py:573–574,656–659`). They are absent from the recorded 51-gate log.

### Q5 · Dark landing

With today's manifest and policy, the fixed grader stays dormant. The legacy freshness check requires caller run context, which WATCHER's current call does not supply (`training_health_check.py:242–250,280–292`, `agents/watcher_agent.py:3182–3190`). Skip expiry and history pruning have independent switches. At the shipped settings the skip reader still checks the count without consulting the clock, and the health archive prune exits before touching history (`training_health_check.py:901–941,1111–1134,1242–1255`). My scratch count-three comparison confirmed the existing skip behavior. Finding 1 records the owner decision that governs a separate benching change.

The new settings are read and validated at legacy (`training_health_check.py:130–178,242–250`). Unknown enum values reject the requested behavior. At the public health boundary the exception becomes `PROCEED_WITH_NOTE` with `severity=absent`, and WATCHER continues, so the fail-closed claim applies to grading and its side effects (`training_health_check.py:223–239`, `agents/watcher_agent.py:2570–2586`). In the compare optimizer an invalid `expiry_mode` with a registry entry raises before dispatch (`meta_prediction_optimizer_anti_overfit.py:156–191`, `training_health_check.py:909–918`). The fixed path can accept an internally consistent old pointer/report pair without WATCHER's run identity, a recorded limit (`training_health_check.py:738–750,841–883`, wave-2 READY note:225–228). I have no objection to this landing under the current settings. The pending owner activation still needs the run-identity and benching decisions accounted for.

## Checks performed

I ran 26 selected existing wave tests on CPU in the `~/venvs/torch` environment, with pytest cache disabled and `--basetemp` under `/tmp`. They all passed. I ran the three baseline-pin probes, rebuilt the optimizer merge, parsed the submitted Python files, compared the saved CatBoost predictions, exercised the compare dispatcher with mocked children, tested the NN argv comparator on a copied trace, and reproduced the out-of-directory write in a disposable repository. `git diff --check` reported six added whitespace-only lines in the submitted optimizer and health-check files. I read Alpha's full-suite logs and summaries, including `54/54` and `4/4` for the landing suite, and did not independently rerun that suite or the full mutant suites. GPU, real-data, and end-to-end WATCHER runs were **UNAVAILABLE** to Beta. The evidence is about diagnostics, dispatch and artifact parity. It does not measure the learning loop's pool hit rate over `k/1000` on unseen draws.
