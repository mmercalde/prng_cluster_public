# X-1B-STEP6-NN · R1 · Team Beta review

## Prior record

- I ran `rg -ni` for scaler, target normalization, relocation, saved Step 6 evidence, and WATCHER failure and freshness behavior across the project catalog, rulings, proposals, chapter audits, consolidated TODO, backlog, leads, and exchange. I also searched the wider documentation for the specific failure mechanisms below, read the local owner notes and the LL-6 and LL-8 exchange, and traced callers with `git grep -n` and `rg -n` in the detached worktree.
- **ALREADY RECORDED** — The written-path scaler gap and the former best-effort behavior are recorded in X-LL6 R1 Q4 and its closing note. Missing checkpoints falling through to feature-mean scoring are Chapter 7 B2. Output reuse by mtime is recorded in the Step 6/Chapter 13 Stage A report. The record needed for Q2 and Q5 is in Finding 1. `docs/exchange/from_beta/X-LL6-BUNDLE_R1_BETA_REVIEW.md:48-65`, `docs/exchange/to_beta/X-LL6-BUNDLE_R2_ALPHA_CLOSING_NOTE.md:131`, `docs/CHAPTER_7_AUDIT.md:200`, `docs/CLAUDE_CODE_REPORT_STEP6_CH13_GRADING_CONTRACT_STAGE_A.md:153-166`.
- **CHANGES A RULING** — Item 1b supersedes Step 6's previous best-effort rule. The local owner note confirms the refusal when the training scaler cannot be found. I used the request's account of the 19:59 direction on target metadata and the 20:33 relocation question. The older Category B proposal recorded compatibility with missing scaler fields. `logs/e2e_session/OWNER_RULING_next_work_20261126.txt:28-30`, `docs/PROPOSAL_CATEGORY_B_NN_TRAINING_ENHANCEMENTS_v1_0.md:303-310`, `docs/exchange/to_beta/X-1B-STEP6-NN_R1_ALPHA_REVIEW_REQUEST_step6_nn_refusal.md:24` and `:49`.
- **NEW** — The exit 4 interface and diagnostic reason are implementation choices. The test results below add evidence on the stored-scale clamp, a checkpoint missing only `y_std`, and the interaction between the refusal and WATCHER's existing freshness check.
- Search limits were live run state, a host-wide checkpoint or launcher inventory, ser8, private remotes, and the rigs and Proxmox hosts. I read Alpha's local development fixtures and logs named below.

**Date** 2026-11-26 PST · **Class** Before commit, at Michael's request · **Base read** `a85adb196adf703c500a4c0cbdfae505d560fe18`

`git status --porcelain` in `~/tfm_worktrees/1b_step6_nn` at the read, rechecked at 20:39 PST. All five SHA-256 prefixes matched the request.

```text
 M models/inference_bundle.py
 M prediction_generator.py
 M scripts/diagnostics/e2e_vote_harness.py
 M tests/test_e2e_vote_harness.py
?? data/
?? tests/test_1b_step6_nn_refusal.py
```

`git status --porcelain` in the main checkout at the read.

```text
?? docs/exchange/to_beta/X-1B-STEP6-NN_R1_ALPHA_REVIEW_REQUEST_step6_nn_refusal.md
```

## Positions

| Question | Position | Severity |
|---|---|---|
| Q1 · Refusal conditions | OBJECTION | THEORETICAL — Step 6 accepts a damaged scaler in a CPU fixture, and I found no evidence that current Step 5 has emitted a negative scale. |
| Q2 · Relocated NN | OBJECTION | LIVE — Step 6 refuses a complete NN when its checkpoint was moved into the supported `models_dir` fallback. |
| Q3 · Input normalization off and pre-S121 | NO OBJECTION | — |
| Q4 · Shared loader | NO OBJECTION | — |
| Q5 · Failure surface | OBJECTION | LIVE — A manual refusal can leave an earlier pool that a subsequent ordinary WATCHER invocation can accept as fresh. |
| Q6 · Tests | NEEDS EVIDENCE | — |

## Findings

### 1. Governing record omitted from the request

**ALREADY RECORDED.** LL-8 landed on the declared base with the owner's checkpoint pin and the X-LL7-LL8 R1 Q3 correction for relocation **after** scoring. Its gate requires saved Step 6 score evidence when a written NN path is missing and verifies that evidence when supplied. System Map §4.2 requires parity with actual Step 6 scores. For WATCHER, R-1 gives an explicit Step 6 failure precedence over output-file validation. Stage A also records the separate pre-dispatch mtime shortcut. These govern Q2 and Q5. `logs/e2e_session/OWNER_RULING_ll78_plan_20261126.txt:2-5`, `docs/exchange/from_beta/X-LL7-LL8_R1_BETA_REVIEW.md:52-60`, `docs/TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md:95-111`, `scripts/diagnostics/attribution_cycle.py:478-508` and `:542-557`, `docs/CLAUDE_CODE_INSTRUCTIONS_WATCHER_SAFETY_REPAIRS_20261111.md:56-75`, `docs/CLAUDE_CODE_REPORT_STEP6_CH13_GRADING_CONTRACT_STAGE_A.md:153-166`.

## Reasoning

### Q1 · Refusal conditions

For a loaded, intact current Step 5 NN on its original path, the constructor requires scaler arrays and target metadata, the scorer applies the input transform before `predict()`, and the wrapper applies the inverse transform. The complete CPU control returned 0 and published a pool. The valid relocated case is in Q2. B2 is a recorded case where the NN never loads and the pool can instead come from the feature mean. `prediction_generator.py:362-398`, `:462-485` and `:864-886`, `models/wrappers/neural_net_wrapper.py:347-361`.

The new finiteness check sees the scale **after** the historical loader rewrites every value below `1e-12` to 1.0. In a copy of the 89-feature Step 5 fixture I set one saved scale to `-Inf`. The loader rewrote it to 1.0, Step 6 returned 0, and the published prediction list differed from the intact control. The largest absolute difference between corresponding positions in the two sorted top-10 score arrays was `1.579e-4`. Current training writes standard deviations, replaces exact zeros before training, and saves the values it used. Negative scales therefore need a damaged checkpoint, and a positive scale below `1e-12` would also be changed even though training used the stored value. The review did not establish how often that positive case occurs. Validate the original scaler for Step 6 before any legacy clamp, with finite, strictly positive, one-dimensional arrays of the schema width. Keep the legacy clamp available for historical replay. The fixture establishes that the transform can change a pool. An equal-`k` walk-forward comparison on unseen draws is needed to measure a hit-rate effect against the active arm and `k/1000`. `models/inference_bundle.py:121-145`, `prediction_generator.py:467-473` and `:868-881`, `train_single_trial.py:525-537` and `:757-774`.

### Q2 · Read from the resolved checkpoint

Use the checkpoint that supplied the weights and the target transform for the input scaler too. The factory already tries the written path and then `models_dir/<basename>`. Step 6 discards the factory's metadata result and opens the written path for the scaler. A byte-identical, complete copy in the fallback location loaded as a model in the CPU fixture and exited 4 before scoring. The resolved bundle from that location matched the complete original bundle exactly across all 41 fixture rows. The checkpoint SHA matched and the maximum absolute score gap was 0.0. The same moved bytes with `as_written` gave a maximum gap of 5.021. Use a single factory resolution and carry that path to Step 6, or obtain the input scaler from the same checkpoint read as the weights. Keep the sidecar's written path available for diagnostics. A refusal loses pool coverage for that attempt. The score comparison measures parity, and any effect on hit rate would need equal-`k` walk-forward grading on unseen draws with refusals included in the coverage accounting. `models/model_factory.py:318-331` and `:372-377`, `prediction_generator.py:362-380`, `models/inference_bundle.py:154-163` and `:270-290`.

LL-8 already accepts the pinned resolved checkpoint with saved evidence that matches the original Step 6 scores. Keep G-Q4-SCALER's conservative behavior for historical or unversioned inputs. A missing written path today does not establish the transform an earlier Step 6 applied. Keep the saved-score check on `as_written` for historical unscaled scoring and retain the mismatch check for a checkpoint moved after scaled scoring. Update the LL-8 wording for current Step 6. In the tests, check current relocated Step 6 against the resolved bundle, both when moved before scoring and when moved before replay of saved scores. Use a historical Step 6 fixture for the positive `as_written` case. Internal A-NORM and the checkpoint pin alone each leave the original scorer's transform unverified. `scripts/diagnostics/attribution_cycle.py:48-64`, `:184-226`, `:483-508` and `:542-557`, `tests/test_ll8_attribution_cycle.py:626-721`, `tests/test_ll6_inference_bundle.py:593-610`.

### Q3 · Input normalization off and pre-S121

Refusal fits the stated production policy. Step 5 forces input normalization for NN comparison and final training. The direct training function can still produce a raw-input NN, and a genuine pre-S121 NN can have a raw target. Those modes can be internally consistent. Current production checkpoints should carry the metadata that Step 5 writes. A missing target field on its own does not identify a checkpoint's era, so the message should describe the missing field without dating the checkpoint. Archive inventory is **UNAVAILABLE**. `meta_prediction_optimizer_anti_overfit.py:280-286` and `:2640-2651`, `train_single_trial.py:493-495`, `:525-553` and `:757-774`, `models/wrappers/neural_net_wrapper.py:355-359`, `prediction_generator.py:474-485`.

### Q4 · Shared loader

The added `reason` is consumed by Step 6. The bundle reads the same three scaler values and passes those same values through to its consumers. Chapter 13's canonical loader, LL-7's bundle handoff, and LL-8's resolved and historical modes retain their current numerical behavior. LL-8 can still refuse a report under its own evidence contract. “Returns in every case” overstates the helper's guarantee. The six named exits return, but conversion and array logging sit outside the `torch.load` handler. For example, the existing `scaler_scale.min()` raises for an empty array. Keep the strict Step 6 validation separate from the legacy replay path or make any shared API change explicitly opt-in. `models/inference_bundle.py:87-145` and `:270-290`, `prediction_generator.py:427-433`, `chapter_13_orchestrator.py:675-690` and `:742-776`, `scripts/diagnostics/attribution_cycle.py:483-508`.

### Q5 · Failure surface

Once dispatched, a real exit 4 is a Step 6 failure inside canonical WATCHER. `run_step` returns `success=False` before loading results, and R-1 escalates before the file-exists evaluation. The Step 6 manifest has no `skip_on_fail` exception. The vote harness also translates `NNNormalisationError` to `HarnessError` during generator construction. The outer WATCHER `--run-pipeline` shell command can still fall through with process status 0 after an escalation because the dispatcher and entry point do not propagate a pipeline status. That is an existing shell-interface limitation. `agents/watcher_agent.py:315-331`, `:1107-1142`, `:1672-1702`, `:2468-2492`, `:3164-3175`, `:3272-3273`, `:4179-4202` and `:4231-4232`, `agent_manifests/prediction.json:197-200`, `scripts/diagnostics/e2e_vote_harness.py:693-703`.

A manual refusal left the earlier canonical pool byte-for-byte in place in scratch. I executed the unchanged WATCHER freshness function against that file and the fixture inputs. It returned `True`. In WATCHER the resulting pre-dispatch branch reports `success=True, skipped=True`. Its manifest dependency list contains the sidecar but omits the checkpoint and the refusal state. This is reachable with an earlier output newer than the declared inputs, including when only checkpoint contents change. The standalone `--evaluate` path also grades a file without that invocation's exit status. Preserve historical output and make reuse of the canonical result conditional on a successful attempt for the current inputs and checkpoint. An mtime-only pass does not establish that. `prediction_generator.py:949-965` and `:1119-1136`, `agent_manifests/prediction.json:6-12` and `:39-42`, `agents/watcher_agent.py:969-1000`, `:1980-2012` and `:4156-4169`.

I also removed each target field separately. Missing only `y_mean` used exit 4. Missing only `y_std` raised `TypeError` in the wrapper's formatting of `None` and used exit 1. It produced no new pool, but bypassed both the Step 6 refusal message and the harness's typed-refusal translation. Make wrapper logging safe for partial metadata so Step 6 can deliver its documented refusal. `models/wrappers/neural_net_wrapper.py:436-444`, `prediction_generator.py:393-409`, `:474-485` and `:1131-1136`, `scripts/diagnostics/e2e_vote_harness.py:699-703`.

### Q6 · Tests and evidence

The current CLI gates delete any old output or use a fresh directory and the missing-target fixture removes both fields together. Add cases for each missing target field on its own, a malformed stored scaler before the legacy clamp, and an earlier canonical prediction after a refusal. Exercise the WATCHER decision with a retained file and explicit child failure, and the later freshness decision independently. The targeted relocation tests need the changes in Q2. I read both Alpha tracebacks and confirmed the failures at `tests/test_ll6_inference_bundle.py:600` and `tests/test_ll8_attribution_cycle.py:642`. The earlier suites stop at those gates and a green run of the revised suites is pending. `tests/test_e2e_vote_harness.py:177-181`, `:203-226` and `:700-749`, `tests/test_1b_step6_nn_refusal.py:146-160`, `:334-360`.

Independent checks used the Torch venv, a previously generated synthetic checkpoint and survivor fixture, and scratch paths under `/tmp/beta-step6-nn.dfJwPE`. I ran the standalone Step 6 entry point on CPU, compared the bundle scores, ran the isolated WATCHER freshness function, and checked `git diff --check`. The unmodified G13 launcher selected CUDA under the CPU-only environment and stopped on device initialization. With Step 6's device selection forced to CPU in the harness and CLI subprocesses, G13 passed and all three mutants M15, M16 and M26 were killed. These runs exercised the refusal assertions on CPU. I read Alpha's CPU suite logs and the affected call sites. Full WATCHER execution, GPU-mode G13 and its mutation suite, and grading on unseen draws are **UNAVAILABLE** in this review. `tests/test_1b_step6_nn_refusal.py:67-74`, `tests/test_e2e_vote_harness.py:203-226` and `:700-786`, `prediction_generator.py:367-371`.
