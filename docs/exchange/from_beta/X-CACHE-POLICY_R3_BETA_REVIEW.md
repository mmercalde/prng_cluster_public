## Prior record

1. **Finding 1 · ALREADY RECORDED, omitted governing references.** R2 finding 1 identified the Route A routing review and pinned-geometry ruling. R3 carries forward the CLI proof indirectly through the Run 4 checklist. Keep the direct references and the operational scope of Run 4. The D6 integration repair record also governs this launcher change. It records the certified SSH detachment and launcher wait-set checks, which the no-torch baseline reported green. `docs/exchange/from_beta/X-CACHE-POLICY_R2_BETA_REVIEW.md:3`, `docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md:23`, `:145`, `docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md:168`, `docs/SESSION_CHANGELOG_20260815_S183.md:65`, `:114`, `docs/CLAUDE_CODE_REPORT_MINER_NO_TORCH.md:173`.
2. **ALREADY RECORDED.** The cache-policy R1/R2 and GCVM reviews establish the landing checks and the first-run cache evidence. The Run 4 checklist, delegated-execution plan and S172 completion plan govern launch, the whole-run evidence and acceptance. `docs/exchange/from_beta/X-CACHE-POLICY_R1_BETA_REVIEW.md:74`, `:97`, `docs/exchange/from_beta/X-CACHE-POLICY_R2_BETA_REVIEW.md:74`, `:87`, `:93`, `docs/exchange/from_beta/X-GCVM-LOCATED_R3_BETA_REVIEW.md:92`, `docs/RUN4_LAUNCH_CHECKLIST.md:348`, `:376`, `:510`, `docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:55`, `:118`, `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:79`, `:317`.
3. **NEW.** The v6 verification, the existing D6 fixture regression and the shallow installed-layout case below. No ruling changes.

Searched the exchange, drafts, session reports, Run 4 checklist, `docs/TB_RULING_*`, `docs/PROPOSAL_*`, chapter audits and `docs/TODO_CONSOLIDATED.md` for the cache-policy requirements, D6 launcher regression coverage, draft-path assumptions, Route A provenance, and launch and acceptance evidence. Searched the existing tests for consumers of the launch scripts. Ser8, private remotes, the Claude project, live ledgers, rigs and Proxmox were outside the search.

# X-CACHE-POLICY · R3 · BETA REVIEW

- Date 2026-11-18 PST.
- Submitted base read `048fd47c92932962c4379c94b787852df8fdb7c7`. Worktree HEAD read `3579088b8e9c2d902252d626baae15bf0647907f`.
- All eight submitted SHA-256 values and all three target SHA-256 values matched. Both repository targets matched the submitted base. All three diffs applied to copies with zero fuzz. The three diffs and verifier matched v5 byte for byte.
- Initial `git status --porcelain`

```text
 M docs/CHAPTER_13_IMPLEMENTATION_PROGRESS_v1_6.md
 M docs/LEADS.md
 M docs/SELFPLAY_ARCHITECTURE_PROPOSAL_v1_0.md
 M docs/TRIANGULATED_FUNCTIONAL_MIMICRY_VERIFIED_v1_0.md
 M docs/exchange/DEFERRED.md
 M docs/exchange/LEDGER.md
 M miner/range_miner_worker.py
 M scripts/launch_fleet_manual.sh
?? docs/CLAUDE_CODE_INSTRUCTIONS_RIG_GPU_FAULT_20261115.md
?? docs/CLAUDE_CODE_PLAN_RIG_ARM_V1_V2.md
?? docs/CLAUDE_CODE_PLAN_RIG_GPU_PROBE.md
?? docs/CLAUDE_CODE_REPORT_GCVM_LOCATED_20261116.md
?? docs/CLAUDE_CODE_REPORT_MINER_NO_TORCH.md
?? docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md
?? docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT_DECODE_20261115.md
?? docs/CLAUDE_CODE_RUNBOOK_RIG_FAULT_RECORDER_AB.md
?? docs/SESSION_CHANGELOG_20261115_RIG_GPU_FAULT.md
?? docs/SESSION_CHANGELOG_20261116_RIG_FAULT_RECORDER.md
?? docs/drafts/
?? docs/exchange/from_beta/X-CACHE-POLICY_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-CACHE-POLICY_R2_BETA_REVIEW.md
?? docs/exchange/from_beta/X-GCVM-LOCATED_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-GCVM-LOCATED_R2_BETA_REVIEW.md
?? docs/exchange/from_beta/X-GCVM-LOCATED_R3_BETA_REVIEW.md
?? docs/exchange/from_beta/X-MINER-NO-TORCH_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-RIG-GPU-FAULT_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-RIG-GPU-FAULT_R2_BETA_REVIEW.md
?? docs/exchange/from_beta/X-RIG-GPU-FAULT_R3_BETA_REVIEW.md
?? docs/exchange/from_beta/X-RIG-GPU-PROBE_R1_BETA_REVIEW.md
?? docs/exchange/to_beta/X-CACHE-POLICY_R1_ALPHA_REVIEW_REQUEST_cache_policy_v4.md
?? docs/exchange/to_beta/X-CACHE-POLICY_R2_ALPHA_REPLY.md
?? docs/exchange/to_beta/X-CACHE-POLICY_R3_ALPHA_REPLY.md
?? docs/exchange/to_beta/X-CUPY-CACHE-RESET_R1_ALPHA_REVIEW_REQUEST_launcher_cache_reset.md
?? docs/exchange/to_beta/X-GCVM-LOCATED_R1_ALPHA_REVIEW_REQUEST_fault_address_reading.md
?? docs/exchange/to_beta/X-GCVM-LOCATED_R2_ALPHA_REPLY.md
?? docs/exchange/to_beta/X-GCVM-LOCATED_R3_ALPHA_REPLY.md
?? docs/exchange/to_beta/X-MINER-NO-TORCH_R1_ALPHA_REVIEW_REQUEST_cleanup_torch_flag.md
?? docs/exchange/to_beta/X-RIG-GPU-FAULT_R1_ALPHA_REVIEW_REQUEST_fault_evidence_and_plan.md
?? docs/exchange/to_beta/X-RIG-GPU-FAULT_R2_ALPHA_REPLY.md
?? docs/exchange/to_beta/X-RIG-GPU-FAULT_R3_ALPHA_REPLY.md
?? docs/exchange/to_beta/X-RIG-GPU-PROBE_R1_ALPHA_REVIEW_REQUEST_torch_arms_probe_controller.md
?? scripts/rig_fault_map.py
?? scripts/rig_fault_recorder.py
?? scripts/rig_gpu_recorder.py
?? scripts/rig_seam_controller.py
?? scripts/rig_seam_probe.cleanup.sha256
?? scripts/rig_seam_probe.py
?? tests/test_miner_no_torch.py
?? tests/test_rig_fault_map.py
?? tests/test_rig_fault_recorder.py
?? tests/test_rig_seam_probe_controller.py
```

| Question | Position | Severity |
|---|---|---|
| Q1 · Landability | OBJECTION | LIVE — the existing D6 launch regression reports failure when its normal stubbed dispatch reaches the v6 default verifier. |
| Q2 · R2 items 1–4 | SUGGESTION | — |
| Q3 · First counted run | SUGGESTION | — |

## Q1 · NOT LANDABLE as submitted

**One landing blocker.** The package breaks the existing D6 integration suite's successful-launch arms. The suite copies the launcher, a worker stand-in, configuration and SSH stub into its private checkout. The v6 default dispatch can succeed and produce the expected ACKs, then the copied launcher refuses because the fixture lacks `scripts/cache_records_verify.py`. The same package removes the old literal assignment required by D6 check WS-7. Both are test compatibility failures. The supplied v6 suite uses a fixture that copies the verifier and exercises both cache policies. `tests/test_s172_d6_integration_repair.py:743`, `:789`, `:984`, `:994`, `:1124`, `:1213`, `docs/drafts/cache_policy_v6/launch_fleet_manual.diff:20`, `:29`, `:95`, `:138`, `docs/drafts/cache_policy_v6/tests/test_cache_policy_v6.py:300`, `:634`.

**Reproducer.** Apply `launch_fleet_manual.diff` to a temporary baseline copy with `patch --fuzz=0`. Load `tests/test_s172_d6_integration_repair.py` without calling its main and use `build_launcher_fixture` with the patched bytes and `ssh_shim=STUB_SSH_EXEC`. With `RIG_CUPY_CACHE_POLICY` unset, set `RIG_CUPY_CACHE_ROOT` to an existing directory inside the fixture. Run its `run_launcher` helper without releasing the stand-ins, then call its `teardown`. In this sandbox I used a dummy port because loopback socket creation was denied. The baseline returned **0**, with two ACKs. v6 returned **1**, with two ACKs and `can't open file .../scripts/cache_records_verify.py`. Explicit `shared` returned **0**, with two ACKs. WS-7's literal `CUPY_CACHE_DIR=/tmp/cupy_cache_gpu$N` was present in the baseline and absent from patched bytes under either policy. `tests/test_s172_d6_integration_repair.py:690`, `:743`, `:789`, `:879`, `:994`, `:1127`, `docs/drafts/cache_policy_v6/launch_fleet_manual.diff:29`, `:95`, `:138`.

Update the D6 fixture and WS-7 in the landing change. One way is to make the D6 stand-ins explicitly exercise the shared diagnostic path, preserve the historical red controls, and check the per-worker path for that policy. Keep the executable v6 default-policy and counted-path checks. Rerun the existing D6 suite and the v6 suite on the resulting tree. `tests/test_s172_d6_integration_repair.py:908`, `:971`, `:1124`, `:1177`, `docs/drafts/cache_policy_v6/tests/test_cache_policy_v6.py:575`, `:653`.

## Q2 · R2 items 1–4

| Item | Review result |
|---|---|
| 1 · Scratch path | **The R2 scratch-path blocker is closed for the intended layout.** The installed test passed with its default scratch under the checkout's `logs/`. Draft-path discovery still runs at import, so the README overstates the independence of landed mode. Copy the test directly to `/tmp/<checkout>/tests/` and invoke it with `--landed /tmp/<checkout> --help`. It raises `IndexError: 2` at `V5.parents[2]` before parsing arguments. This shallow placement is outside the recorded VM101 production checkout and worktree paths. Move the path calculation into pre-landing setup and correct the README. `AGENTS.md:151`, `docs/drafts/cache_policy_v6/README.md:46`, `docs/drafts/cache_policy_v6/tests/test_cache_policy_v6.py:31`, `:58`, `:790`. |
| 2 · Footprint timeout | **Holds.** The timed-out row reports endpoint, path and timeout in its reason, the healthy row is measured, and the collector writes `INCOMPLETE` and exits 1. The case and mutant F2 reproduced. `docs/drafts/cache_policy_v6/scripts/cache_footprint.py:42`, `:67`, `:82`, `docs/drafts/cache_policy_v6/tests/test_cache_policy_v6.py:559`. |
| 3 · Provenance description | **Holds.** Both entry-point diffs hash the entry point and the five listed scripts and record HEAD. The clean committed revision identifies the footprint script. The README now describes the individual digests and the separate post-run command accurately. `docs/drafts/cache_policy_v6/README.md:48`, `docs/drafts/cache_policy_v6/gate12_launch.diff:41`, `:101`, `docs/drafts/cache_policy_v6/run4_launch.diff:41`, `:101`. |
| 4 · Encoder status and CT-side filter | **Holds.** The probe reads both pipeline statuses, emits rc 90 if the encoder fails, and sends NUL-framed entries with the requested variable name. The join refuses the missing, duplicate and wrong values and the rc 2/90 cases. The non-UTF-8 control passed and mutants J7 and J8 were detected. `docs/drafts/cache_policy_v6/scripts/gate12_cache_join_gate.py:53`, `:209`, `:217`, `docs/drafts/cache_policy_v6/tests/test_cache_policy_v6.py:488`, `:512`. |

## Q3 · First counted run

The README covers the earlier cache-policy evidence items. Make the following corrections before using it as the complete checklist for a counted result.

1. **Correct the Route A comparison.** Compare the installed external wrapper with the **landed `gate12_launch.sh`** and require the diff to be limited to `--params`. The submitted patched wrapper is byte-identical to the shipped Route A fixture, which already has the Run 4 parameter block. I verified that equality on the patched copy, the parameter-only diff against patched Gate 12, and all 17 values and types against `RUN4_PARAMS`. Retain the installed digest, landed test, and committed-tree Route A provenance check as separate evidence. `docs/drafts/cache_policy_v6/README.md:58`, `docs/drafts/cache_policy_v6/tests/fixtures/run4_launch.sh:465`, `docs/RUN4_LAUNCH_CHECKLIST.md:320`, `:348`, `:376`.
2. **Carry the full Run 4 checklist and acceptance steps forward by reference.** This includes Michael's authorization, infrastructure checks and netconsole proof, the volume gate, and the pinned search open/status check. Keep clean committed launch, GPU and parity admission, and committed-HEAD deployment with 30/30 verification if governed bytes change. Before accepting Run 4, require a completed run and publication, run-scoped `G-PROD-SHAPE` PASS, search and coverage confirmation, the hashed whole-run evidence and closure report, and evidence review before acceptance unless Michael decides directly. Verify the cache freeze's `SHA256SUMS` and the 24 rig joins with Zeus recorded separately as part of that evidence. For a delegated run, include the append-only session record. `docs/RUN4_LAUNCH_CHECKLIST.md:89`, `:299`, `:304`, `:337`, `:510`, `:533`, `:552`, `docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md:145`, `scripts/gate12_parity_gate.py:174`, `docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:55`, `:118`, `:175`, `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:55`, `:79`, `:317`.
3. **Arrange collection of the timing sources before dispatch.** Record join and freeze start and end on VM101. Preserve the 25 workers' nonce-matched park, release and timeout evidence before subsequent dispatches reuse the logs, including remote `/tmp/minerlogs/gpu<N>.log`. The pre-release freeze already contains the 24 startup measurements and liveness evidence for Zeus. It precedes the release events. Report the configured stagger, elapsed time from first park to release, maximum successful worker-local `waited_s` against the 900-second deadline, any timeouts, and missing worker evidence. The 480 seconds describe the 24 SSH timeout budgets. Freeze time, other gates and coordinator bind also consume parking time. Retain and hash the separate footprint JSON in the whole-run bundle. An `INCOMPLETE` collection leaves the footprint evidence outstanding. Live timing and footprint remain **UNAVAILABLE**. `docs/drafts/cache_policy_v6/README.md:70`, `:77`, `docs/drafts/cache_policy_v6/gate12_launch.diff:61`, `:76`, `:83`, `:101`, `docs/drafts/cache_policy_v6/scripts/gate12_cache_join_gate.py:48`, `:172`, `scripts/launch_fleet_manual.sh:223`, `:335`, `gate12_launch.sh:398`, `miner/range_miner_worker.py:2202`, `:2214`, `:2223`.

## Checks and limits

Activated `~/venvs/torch` and ran the submitted suite on temporary copies. Pre-landing **112/112**, installed under `tests/` with the patched external Route A copy **111/111**, and **22/22 mutants detected**. Both entry-point slices checked the join and freeze before the sampler marker, including refusal cases. I also ran the focused D6 reproduction, shallow path check, hash and patch checks, and parameter comparison above. Full D6 execution with its real loopback listener is **UNAVAILABLE** in this sandbox. The committed-tree Route A provenance check, full live entry points, rig results and held-out hit rate are **UNAVAILABLE** for this review. I checked the submitted inputs against their initial hashes before writing the review. Michael decides landing and run acceptance.
