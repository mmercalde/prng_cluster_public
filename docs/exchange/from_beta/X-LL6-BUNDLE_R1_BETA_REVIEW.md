# X-LL6-BUNDLE · R1 · Team Beta review · 2026-11-25

## Prior record

I repeated the request's searches with `rg -n -i` over `docs/`, `CLAUDE.md`, and the named task logs for `inference_bundle|inference bundle|c13[-_]model[-_]loader|A-NORM|Blocker 3|LL-6`. I searched the rulings, proposals, audits, consolidated TODO, drafts and exchange for `ranked.prediction`, `positional.alignment`, `tier.compar`, `wrong.checkpoint`, `MPLCONFIGDIR` and `no_write_outside_out` before recording findings. I checked the named landing notes and the W2 and block 0 records. The ser8 archives, private remote, rigs, and live run state were outside these searches.

- **ALREADY RECORDED** — The system map §§4.2, 4.5, P0-C and §14 govern the loader, A-NORM and the ranked-record contract. Chapter 13 audit B15/S6, Chapter 14 audit S2, and consolidated TODO P0-C record the activation limits. `TASK_LOOP_ADDENDUM_3.txt:14` and `TASK_LL6_LAND_PREP_20261125.txt:2-8` set the landing scope. The 17:45 owner order governs the later switch registration. Finding 1 adds the omitted owner direction and the scope of the branch's real-model evidence.
- **ALREADY RECORDED** — The W2 record documents the relocated NN scaler mismatch. The request and READY note already record NN training mode on return and the named-row tier failure. The system map §4.5 already prohibits unverified positional alignment.
- **NEW** — Parity on this landing's code with the four archived block 0 models, the captured NN attribution forward, the no-write false pass, the LightGBM CLI false failure, and the surviving Chapter 13 handoff mutant.
- **CHANGES A RULING** — None.

I read detached base `7f4505aaccebdd284393eb3565e75d09d1c0f3b4` at `~/tfm_worktrees/ll6-land`. Main was `579dbb7e558da1d90032967ca7b3e28054180e7a`. The cross-base diff touched seven files and none of the five landing paths. All five source hashes and the READY note hash matched the request on initial read and recheck.

`git status --porcelain` in the landing worktree at initial read:

```text
 M chapter_13_orchestrator.py
 M prediction_generator.py
?? data/
?? docs/drafts/loop_v1/POSTLAND_CHECK_579dbb7_20261125.md
?? docs/drafts/loop_v1/READY_LAND_LL6_20261125.md
?? tests/ll6_step6_runner.py
?? tests/test_ll6_inference_bundle.py
```

## Positions

| Question | Position | Severity |
|---|---|---|
| Q1. Default behavior | NO OBJECTION | — |
| Q2. Canonical loader and A-NORM for LL-6 | NO OBJECTION | — |
| Q3. NN mode and tier comparison | SUGGESTION | — |
| Q4. Scaler source | NO OBJECTION | — |
| Q5. Tests after landing | OBJECTION | LIVE — A CPU replay on VM 101 created a cache directory outside `--out` while the no-write gate passed. |

## Q1. Default behavior

The scaler starts disabled in Step 6. The refactor preserves the checkpoint read, best-effort branches, float32 arrays, `1e-12` clamp, and logger on the normal entry point. NumPy was already imported at module level before LL-6. WATCHER invokes the Step 6 script, and walk-forward invokes the vote harness that constructs the same generator. The legacy Chapter 13 dispatch and both legacy attribution input conversions preserve the baseline route. `7f4505a:prediction_generator.py:44`, `:403-454`, landing `prediction_generator.py:287-290`, `:354-370`, `:401-406`, `:783-799`, `models/inference_bundle.py:68-119`, `agents/watcher_agent.py:412`, `:2354-2361`, `scripts/diagnostics/e2e_walk_forward.py:1193-1210`, `scripts/diagnostics/e2e_vote_harness.py:687-697`, `chapter_13_orchestrator.py:127-138`, `:586-589`, `:746-775`.

I verified equal scaler flags, arrays and three log lines. All four CPU CLI fixtures returned 0 and matched on the 35 stable output keys, with 10 predictions each. The baseline repeat isolated the two changing JSON timestamps. The legacy LightGBM root-cause fixture matched apart from its timestamp, and the test reported the same legacy loader summary for each family. After setting `MPLCONFIGDIR`, all runner-captured log lines matched the baseline and control for all four families. WATCHER execution, GPU output parity and a walk-forward fingerprint comparison are **UNAVAILABLE** as independent Beta measurements. This is enough evidence for the scoped default refactor. `tests/test_ll6_inference_bundle.py:377-429`, `:470-514`, `:539-557`.

## Q2. Canonical loader and A-NORM

Chapter 13 canonical loading uses the same Step 5 sidecar factory as Step 6. It orders named rows from the sidecar and uses the same NN scaler reader. For LL-6, an independent forward through the attribution object in eval mode, mapped back to target units for the NN, measures the prediction A-NORM specifies. I also captured the actual Chapter 13 NN backend's 20 forwards on the fixture. They were all in eval mode and differed from the corresponding Step 6 scores by at most `9.38e-9`. `prediction_generator.py:354-370`, `models/inference_bundle.py:166-195`, `:200-215`, `:228-264`, `chapter_13_orchestrator.py:89-99`, `:674-689`, `:746-775`, `per_survivor_attribution.py:84-123`, `models/wrappers/neural_net_wrapper.py:350-359`, `tests/test_ll6_inference_bundle.py:167-194`.

I used this landing's Step 6 and Chapter 13 canonical loader on all **12,156** saved Step 3 rows from development block 0 and its four archived models, read-only. Step 6 scores matched the archived CPU scores exactly, and each loaded checkpoint hash matched its frozen copy. The Chapter 13 input transform plus attribution-object forward differed by `7.46e-9` for NN and `0` for XGBoost, LightGBM and CatBoost at `atol=1e-5`. The synthetic suite gave `7.74e-9` and three zeros on the same tolerance. An end-to-end Chapter 13 draw-to-artifact cycle on this landing is **UNAVAILABLE** as Beta evidence. That gate still requires the artifact contract, checkpoint and schema identity, and a real observe-only cycle. Neither parity check measures hit rate against `k/1000`. See system map §§4.2, 4.5 and P0-C, and Finding 2.

## Q3. NN mode and tier comparison

Eval mode at bundle load is optional for LL-6. Both current consumers call eval before forwarding, and the fixture hook confirmed Chapter 13 restored training mode after its successful calls. Put the tier fix with LL-7 or LL-8 before relying on tier output. The reproduced named-row fixture archived the root cause with 18 miss and two hit attributions and logged the dict conversion failure, leaving no tier file. An equal-width positional NN row can also reach that tier path without scaling. The same validated vector and scaler should feed both paths. Use named rows in a regime-shift test that checks tier output. `models/wrappers/neural_net_wrapper.py:350-353`, `:422-444`, `per_survivor_attribution.py:90-121`, `:375-390`, `chapter_13_orchestrator.py:840-851`, `:859-904`.

## Q4. Scaler source

Preserve Step 6's current resolution in LL-6. The factory can load the relocated checkpoint while Step 6 still uses the path written in the sidecar for its scaler. Canonical uses the resolved checkpoint. On my fixture the maximum score gap was `4.55` with `resolved` and `0` with `as_written`. Alpha measured `1.89` and `0` on its independently trained fixture. A-NORM therefore has a known failure when the written NN path is missing and Step 6 used the fallback. Apply the parity gate to the actual inference paths before interpreting such attribution. These measurements establish a score difference. Hit-rate impact is unmeasured. `models/model_factory.py:318-330`, `prediction_generator.py:349-366`, `models/inference_bundle.py:128-137`, `:244-264`, `tests/test_ll6_inference_bundle.py:451-468`.

## Q5. Tests after landing

- **Baseline and mutation isolation.** The fixed full SHA and source precondition remain meaningful after landing. The in-process import-origin checks and the runner's early mutant package import cover the path precedence issue. A killed mutant needs every named target explicitly red, plus a marker where declared. A crash with no summary gets no credit. The unmarked constructor mutant proves the signature default and a manually seeded loader dispatch, since the test uses `__new__` and fills in the setting itself. For provenance, record the actual baseline override in the summary or remove the override. The summary currently writes the default constant. `tests/test_ll6_inference_bundle.py:202-239`, `:271-296`, `:311-349`, `:364-375`, `:471-488`, `:508-514`, `:606-610`, `tests/ll6_step6_runner.py:42-65`.
- **LIVE no-write gate.** The lead's task requires writes to stay within the requested output directory. The guard compares Git status after excluding **all of `data/`**. A sibling sentinel outside the selected `--out` was invisible. In the fresh replay, all three LightGBM CLI calls logged creation of distinct `/tmp/matplotlib-…` directories because the default cache path was unwritable. The guard still passed. All four prediction JSON comparisons passed on stable keys, while LightGBM's incidental warning failed the CLI gate. The runner inherits the cache environment and captures third-party warnings, and the CLI gate checks baseline repeatability for JSON only. The initial 11/11 mutant count alone could not establish the three CLI kills because the unmutated gate was already red. In the controlled rerun I verified each of those three mutants changed stable NN prediction JSON keys. Put cache and temp roots under the suite's output before imports and children, using the existing walk-forward pattern. Check the exact output subtree boundary and add a sibling-write negative control. A before/after status comparison alone cannot detect temporary files that are removed during the run. In the CLI check, compare the baseline logs as a control as well, with explicit handling of any expected volatile fields. `logs/e2e_session/TASK_LL6_LAND_PREP_20261125.txt:5`, `tests/test_ll6_inference_bundle.py:137-150`, `:410-429`, `:589-601`, `tests/ll6_step6_runner.py:45-69`, `scripts/diagnostics/e2e_walk_forward.py:388-396`.
- **Missing handoff and checkpoint negatives.** I replaced the orchestrator's handoff at `chapter_13_orchestrator.py:365` with `bundle=None` in scratch. The A-NORM, canonical loader and canonical root-cause gates all stayed green. The root-cause test supplies its own bundle, bypassing the handoff. Add a fixture that exercises the real handoff and kills that mutant. Carry forward W2's wrong-checkpoint A-NORM mutant for a same-family, same-width alternate checkpoint. `tests/test_ll6_inference_bundle.py:431-449`, `:515-535`, `:559-579`, W2 READY note `:151`.

## Additional findings

1. **Prior record omits the owner's learning-loop direction.** `logs/e2e_session/OWNER_DIRECTION_learning_loop_team_20261123.txt:2-6` authorizes the Chapter 13/14 why-work and makes the next graded block the learning loop's test. The cited survey already names that direction. Also cite `TASK_LEARN_WHY_BLOCK0_20261124.txt:2-5` and `READY_WHY_BLOCK0_20261124.md:120-129` for the earlier, development-only real-model A-NORM on the branch. The system map's §4.5 is material to Q2 and should appear alongside the cited loader and scaler sections.
2. **THEORETICAL — Canonical input identity before activation.** The current daemon would need a ranked-record producer and a regression trigger that supplies the expected inputs. The missing trigger inputs and artifact producer are recorded in Chapter 13 audit B15. The new canonical adapter accepts positional vectors by width alone, and the reader returns only rows after an optional draw-ID check and a sample check for `features`. The analysis does not bind the record to the bundle schema or checkpoint. With a same-width reordered vector or an artifact from a different checkpoint, the report could describe the wrong inputs. Enforce the system map §4.5 contract at the canonical boundary before activating it on real artifacts. `chapter_13_orchestrator.py:89-99`, `:343-365`, `:569-582`, `:938-980`, `chapter_13_diagnostics.py:599-667`, `:862-879`.

## Checks run and evidence read

On VM 101 I ran the LL-6 suite in a scratch clone with the five reviewed files, the torch venv, CUDA hidden and CPU library thread limits. The fresh fixture replay gave **11/12 gates, 11/11 mutants** with the LightGBM log false failure. A repeat using the same fixtures and `MPLCONFIGDIR` under its output directory gave **12/12 gates, 11/11 mutants**. I also ran the sibling-write probe, the handoff mutant, the actual NN forward hook, and the CPU score comparisons on archived block 0 outputs. I read Alpha's READY, LL-6 summary and log, negative precondition log, ancillary suite summaries, and registry post-landing note. The other suites in Alpha's table were read evidence. Scratch logs are under `/tmp/beta-ll6.Ssuzsi/`.
