## Prior record

- I used `rg -n -i` and `rg -l -i` for CD-05–07, run identity, checkpoint and report digests, binding, pending publication, fallback, compare pointers, repeated flags, attachments and stale policy. I searched the project file catalog, Beta rulings and ruling requests, proposals, chapters and audits, consolidated TODO, backlog, leads, exchanges and drafts. I checked the R1 review, both W3 designs, the W3 READY record, the G-2 closeout, the W2 review and closing note, and the local owner defaults and registration records. Scope excluded the rigs, host and Proxmox, ser8 archives, private remote and live run state.
- **Finding 1 · ALREADY RECORDED.** The request refers to a fixed reason set without citing the W1 decision that fixes it. D-2 places an unusable checkpoint under `no_checkpoint` and D-5 fixes the reason vocabulary. The CD-07 record also explicitly allows `stale_report_action: grade`, exercised in the G-2 closeout that Alpha cites. That policy matters to the scope of the owner's newer strict guard. The November 23 defaults ruling conditions both fixed grading and full diagnostics on their later landing. `docs/exchange/to_beta/X-W1-RUNNER_R1_ALPHA_REVIEW_REQUEST_why_report_runner.md:15`, `docs/drafts/loop_v1/READY_LOOP_DIAG_2_20261123.md:115–120,132–135`, `docs/drafts/loop_v1/X-BENCHING-SWITCH_R2_ALPHA_CLOSING_NOTE_benching_switch.md:65–73`, `logs/e2e_session/OWNER_RULING_defaults_20261123.txt:5–10`.
- **ALREADY RECORDED.** Chapter 6 B2's compare output routing, legacy's tree-report coverage, the postponed parallel-tree work, the runner's attachment race and the staging retention question. The R1 public-file provenance and startup selection objections are the regressions this review checked. `docs/CHAPTER_6_AUDIT.md:179–186`, `docs/drafts/ch14_v1/PLAN_CH14_DIAGNOSTICS.md:173–187`, `docs/TODO_CONSOLIDATED.md:168`, `docs/exchange/from_beta/X-W2-HOOK_R2_BETA_REVIEW.md:74–76`, `docs/drafts/unit_w/READY_W3_20261127.md:311–335`.
- **CHANGES A RULING.** Identity on the original flag line and the tree pair in the writer's sidecar are the owner's W3 amendments. **NEW.** The fallback publication leak, fixed-grader report and pointer rereads, the strict guard's interaction with the recorded policy, and the no-ID decode exception difference. `docs/exchange/to_beta/X-W3-BINDING_R2_ALPHA_REVIEW_REQUEST_flag_model_binding.md:23–34`.

# X-W3-BINDING R2 · Team Beta code review

2026-11-27, Pacific. Base read in both checkouts `14815a794a23f1872e478c5bd202d9ecd411f74b`. Implementation read in `/home/michael/tfm_worktrees/w3_binding_design`. WATCHER line numbers for the new code refer to the owner patch applied in Beta's temporary checkout.

`git status --porcelain` at intake in main

```text
 M docs/exchange/LEDGER.md
?? docs/exchange/from_beta/X-W3-BINDING_R1_BETA_REVIEW.md
?? docs/exchange/to_beta/X-W3-BINDING_R1_ALPHA_REVIEW_REQUEST_flag_model_binding.md
?? docs/exchange/to_beta/X-W3-BINDING_R2_ALPHA_REVIEW_REQUEST_flag_model_binding.md
```

`git status --porcelain` at intake in the review worktree

```text
 M agents/registry/parameter_registry.py
 M meta_prediction_optimizer_anti_overfit.py
 M scripts/diagnostics/why_report_runner.py
 M tests/test_ch14_diag_wave1.py
 M tests/test_why_report_auto.py
 M tests/test_why_report_flags.py
 M train_single_trial.py
 M training_diagnostics.py
 M training_health_check.py
?? data/
?? docs/drafts/unit_w/BETA_REQUEST_DRAFT_X-W3-BINDING_R1.md
?? docs/drafts/unit_w/BETA_REQUEST_DRAFT_X-W3-BINDING_R2.md
?? docs/drafts/unit_w/DESIGN_FLAG_MODEL_BINDING_20261127.md
?? docs/drafts/unit_w/DESIGN_FLAG_MODEL_BINDING_20261127_REV2.md
?? docs/drafts/unit_w/READY_W3_20261127.md
?? docs/drafts/unit_w/W3_OWNER_LAND.patch
?? tests/test_flag_model_binding.py
```

Git reported a symlink-loop warning for `data/w3_fix1/real_gpu/compare/tree/.gitignore`. The reviewed source, tests and patch were accessible and their digests matched READY §§5 and 10 on both reads.

## Positions

| Question | Position | Severity |
| --- | --- | --- |
| Q1 | NO OBJECTION | — |
| Q2 | OBJECTION | THEORETICAL — Publishing the rejected NN pair requires the optional single-model CLI fallback, which the installed WATCHER manifest does not pass. |
| Q3 | OBJECTION | THEORETICAL — The winner-report reread requires a fixed-grading full compare and an overlapping write to the public family report. |
| Q4 | OBJECTION | THEORETICAL — The no-ID change needs a report with invalid text encoding, which the normal writers do not emit. |
| Q5 | OBJECTION | THEORETICAL — The pointer reread requires a fixed-grading full compare with an overlapping pointer write, and the separate stale-policy bypass requires the nondefault `grade` policy. |
| Q6 | NO OBJECTION | — |
| Q7 | NO OBJECTION | — |

### Q1 · Pair provenance

Different dispatches get different staging paths. Trees build the report from the fitted object they then serialize to the same model's staging directory. The NN worker writes and hashes the staged checkpoint before stamping the report. The NN parent's public rename is outside the digest inputs. The parent recomputes the staged digest and requires the worker report's embedded digest to agree. The existing interleaving gate passed. This establishes the pair's provenance for the WATCHER-issued IDs. Anyone manually starting two writers with the same explicit run/model ID would reuse a directory. Keep one writer per key if direct callers or retries ever reuse IDs. `training_diagnostics.py:134–139,159–184,211–237,676–698`, `meta_prediction_optimizer_anti_overfit.py:1344–1412,1931–1953,2744–2759,2893–2906,3026–3039`, `train_single_trial.py:775–827`, `tests/test_flag_model_binding.py:585`.

### Q2 · Publication and fallback

Optuna folds lack the binding flags and the final-model save. A failed compare child's pending record stays unpublished, and the top-level degenerate exit precedes publication. The post-training NN fallback is the exception. The NN parent writes the pending pair **before** validating the worker's metrics JSON. If that validation raises and the single-shot CLI allows inline fallback, Step 5 can finish successfully with the inline checkpoint and still publish the rejected subprocess's pending pair. In scratch I injected a worker exit 0 with staged files and missing required metrics keys. The real fallback save and `main` completed at exit 0. The published digest differed from the public inline checkpoint, and the real health check raised a bound flag for the rejected subprocess. Move pending creation after NN acceptance and have success publication consume only a pair from the accepted final path. Gate the fallback case with exit 0 and no published pair. `meta_prediction_optimizer_anti_overfit.py:305–328,1330–1334,1931–1953,1976–2009,2754–2759,2774–2801,2934–2969,3356–3357,3485–3501,3522–3533`, `agent_manifests/reinforcement.json:173–187`, `tests/test_flag_model_binding.py:725`.

### Q3 · Exact bytes used for fixed grading

The fixed grader loads and grades the winner and then loads its path again in the per-family loop. The second load overwrites `_w3_loaded[path]` before freshness is checked. In scratch, run A's critical XGBoost winner set the grade. Run B replaced that public file before the per-family load. The family pass correctly called B's report stale, yet the A flag carried B's report digest and came back unbound even though A's published pair remained intact. Save the identity tuple alongside each parsed report at the point of grading. Use the first winner tuple when the winner sets the grade and use the promoted family's own tuple when it sets the grade. Reusing the original winner read in the family loop would also keep its two views consistent. The existing report-swap gate exercises the legacy path, and the promotion gate uses stable files. `training_health_check.py:840–873,876–903,1314–1327,1353–1376`, `tests/test_flag_model_binding.py:309–325,345–358`.

The binding checks' order is sensible once the grade supplies the correct tuple. Refuse ambiguous matches. For future multiple models per family, make the model ID part of candidate selection, verify agreement with the directory and require one record after matching the run, family, model and report digest. Current writers stamp the model ID so their reports are distinguishable even when their metrics match. The checkpoint hash remains last. `training_diagnostics.py:224–235,676–687`, `training_health_check.py:1353–1398`.

### Q4 · No-ID parity

The tested legacy, fixed, promotion and pointer-only stale cases passed against the base. The compare gate checks child argv and resulting files, the tree gate checks checkpoint and sidecar, and the NN worker gate checks report keys and absence of staging. Exact parity has a narrow exception. Both graders now catch `UnicodeDecodeError` on the old read route. An invalid UTF-8 fixture gave `PROCEED_WITH_NOTE / absent` on 14815a7 and `PROCEED_WITH_NOTE / warning` with W3 in both graders. Confine the new exception handling to the binding route and include the corrupt-byte fixture in `W3.absent_thc`. `training_health_check.py:267–280,310–326,840–845`, `tests/test_flag_model_binding.py:873–957`.

### Q5 · Strict guard and pointer authority

An older pointer should be ignored. The canonical report can then establish freshness on its own, as the current leftover-pointer gate demonstrates. The implementation validates one read of the pointer and routes with a second read. I replaced pointer A with pointer B between the reads, leaving A's family reports available. Fixed grading selected B's CatBoost winner instead of A's healthy XGBoost winner and raised a bound CatBoost flag on A's bytes. Keep the one parsed pointer that passed the ID check and use that object for routing. Gate replacement immediately after the pointer read. `training_health_check.py:807–831,846–854,950–966`, `tests/test_flag_model_binding.py:328–342`, `meta_prediction_optimizer_anti_overfit.py:383–394`.

An unstamped report is marked stale at the freshness layer. The recorded `stale_report_action: grade` still permits it to drive the health action. With that policy, the real legacy checker graded an unstamped critical report, advanced the critical count and appended an unbound flag. The request presents the owner decision as a strict guard. Under that reading, use `PROCEED_WITH_NOTE` for this policy conflict when WATCHER supplies an ID, or have Michael explicitly retain the stale-grading override for binding calls. Preserve the no-ID `grade` behavior. The tracked defaults use legacy grading and leave `stale_report_action` at `PROCEED_WITH_NOTE`. `training_health_check.py:83,111,160–174,188–194,332–342,849–853,950–961,986–998,1182–1214`, `docs/exchange/to_beta/X-W3-BINDING_R2_ALPHA_REVIEW_REQUEST_flag_model_binding.md:23–25`.

### Q6 · Runner

`no_checkpoint` with `FlagUnbound` and the recorded reason gives callers the distinction and preserves the W1 reason contract. For a W3 line already present at selection, the newest occurrence controls and an unbound occurrence stops before checkpoint or block selection. A bound digest reaches block selection and the real spec requires that digest. REV2 §4.6 still says `flag_unbound`. Update that wording to the implemented contract. `scripts/diagnostics/why_report_runner.py:98–118,206–233,258–274,539–545,590–606`, `tests/test_flag_model_binding.py:793–861`, `docs/drafts/unit_w/DESIGN_FLAG_MODEL_BINDING_20261127_REV2.md:168–176`.

**ALREADY RECORDED status, reproduced with W3 inputs.** I appended an unbound repeat while a stub LL-8 was returning its successful report for the earlier bound occurrence. The real runner attached the earlier report and closed the group. The occurrence decision is a startup snapshot, and the attachment still checks only group openness and the report hash. Close this recorded race with a shared append/attach guard and occurrence comparison if the contract must hold through attachment. `scripts/diagnostics/why_report_runner.py:590–604,632–666`, `why_report_flags.py:219–249`, `docs/exchange/from_beta/X-W2-HOOK_R2_BETA_REVIEW.md:74–76`.

### Q7 · Owner patch and test changes

The patch creates the ID after freshness skips and overrides the caller ID before argv construction. It stamps the nonzero return, loaded result, bare success, timeout and exception. The installed gate checks dispatch-to-health agreement, retry IDs, freshness skip, caller override and the no-ID reset. The W2 mutant still removes the recorded models-directory route and is caught. The owner-cohort gate checks the installed WATCHER against W2 with the archived W3 patch applied, which supports the shorter marker check. The argv test accounts for the reviewed W3 additions and compares the other builder lines. The wave-1 archive edit takes the `aefdd74` approach, retains each archive for the existing content comparison and also preserves the consecutive-call check. `docs/drafts/unit_w/W3_OWNER_LAND.patch:5–39,43–110`, `agents/watcher_agent.py:2006–2013,2436–2452,2495–2530,2601–2605,3210–3222` (patched), `tests/test_flag_model_binding.py:1052–1113`, `tests/test_why_report_auto.py:154–159,950–995,1127–1144`, `tests/test_why_report_flags.py:389–434`, `tests/test_ch14_diag_wave1.py:153–185,204–212`, `tests/test_benching_switch.py:313–325`.

### Verification

I applied the owner patch in a temporary clone, after `git apply --check`. The installed WATCHER and manifest digests matched READY §4. With the Torch environment active and CUDA disabled, `tests/test_flag_model_binding.py`, `tests/test_why_report_flags.py` and `tests/test_ch14_diag_wave1.py` passed **115/115** together. The standalone `tests/test_why_report_auto.py` harness passed **23/23** gates and killed **21/21** mutants. The scratch probes ran the real readers, publisher, flag appender and runner control flow with synthetic files and stubbed external work where stated. I read Alpha's GPU compare evidence for a bound CatBoost flag with a restored NN winner and the explicit staged-checkpoint lookup after the public file changed. Independent GPU reproduction, a real WATCHER pipeline dispatch and cold-cache timing were **UNAVAILABLE**. Pool hit rate was unmeasured.
