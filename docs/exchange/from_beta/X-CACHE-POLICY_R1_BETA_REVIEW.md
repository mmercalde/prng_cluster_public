## Prior record

1. **ALREADY RECORDED — governing references omitted.** The request names the R3 review but omits the governing records that R3 finding 1 identifies. The delegated-execution plan governs the clean committed launch, parity, evidence preservation and acceptance review. The S172 completion plan retains those launch and acceptance requirements. The findings-resolution proposal reserves the launcher and Gate 12 files from unrelated findings packages. The Run 4 checklist governs the external Route A copy and its parameter-only diff. Carry those direct references forward. `docs/exchange/from_beta/X-GCVM-LOCATED_R3_BETA_REVIEW.md:3`, `docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:55`, `:118`, `:175`, `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:79`, `:317`, `docs/PROPOSAL_FINDINGS_20261113_RESOLUTION_v1_1.md:24`, `:28`, `docs/RUN4_LAUNCH_CHECKLIST.md:25`, `:376`.
2. **ALREADY RECORDED.** R2 records no objection to the fresh on-disk cache per ROCm worker per counted launch. R3 records the v3 parser and write failures and the required landing and run evidence. L-7 records the phase-4 observations on rrig6600 and rrig6600c, cold 8/8 clean and warm 4/4 faulted. `docs/exchange/from_beta/X-GCVM-LOCATED_R2_BETA_REVIEW.md:87`, `:89`, `docs/exchange/from_beta/X-GCVM-LOCATED_R3_BETA_REVIEW.md:72`, `:82`, `:92`, `docs/LEADS.md:590`, `:607`.
3. **NEW.** The v4 implementation and the reproduced Git-provenance, environment parsing and post-rename publication cases below. No change to a ruling.

Searched the exchange, drafts, L-7, GCVM report, `docs/TB_RULING_*`, `docs/PROPOSAL_*`, chapter audits and `docs/TODO_CONSOLIDATED.md` for CuPy cache policy, the counted launch paths, provenance and freeze requirements, and the failure cases below. Compared the submitted diffs with the launch targets, the sentinel and liveness interfaces, and the parity scope. Ser8, the private remote, the Claude project, live rigs, Proxmox and live ledgers were outside this search.

# X-CACHE-POLICY · R1 · BETA REVIEW

- Date 2026-11-17 PST.
- Submitted base read `048fd47c92932962c4379c94b787852df8fdb7c7`. Worktree HEAD read `3579088b8e9c2d902252d626baae15bf0647907f`. Local `main` read `dd43c8b5c3c6791e4be4b21a2c6c70f82e720966`.
- All six submitted SHA-256 values and all three target hashes matched. The repository targets matched the submitted base and local `main`. The three diffs applied with zero fuzz to temporary copies and all three passed `bash -n`. The patched Route A copy differed from the patched repository harness only in the existing `--params` block. `docs/exchange/to_beta/X-CACHE-POLICY_R1_ALPHA_REVIEW_REQUEST_cache_policy_v4.md:35`, `:43`, `docs/RUN4_LAUNCH_CHECKLIST.md:376`.
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
?? docs/exchange/from_beta/X-GCVM-LOCATED_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-GCVM-LOCATED_R2_BETA_REVIEW.md
?? docs/exchange/from_beta/X-GCVM-LOCATED_R3_BETA_REVIEW.md
?? docs/exchange/from_beta/X-MINER-NO-TORCH_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-RIG-GPU-FAULT_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-RIG-GPU-FAULT_R2_BETA_REVIEW.md
?? docs/exchange/from_beta/X-RIG-GPU-FAULT_R3_BETA_REVIEW.md
?? docs/exchange/from_beta/X-RIG-GPU-PROBE_R1_BETA_REVIEW.md
?? docs/exchange/to_beta/X-CACHE-POLICY_R1_ALPHA_REVIEW_REQUEST_cache_policy_v4.md
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
| Q1 · Landability | OBJECTION | LIVE — a Git read failure after clean-tree admission can leave either counted entry point dispatching and proceeding with incomplete launch provenance. |
| Q2 · R3 checklist | OBJECTION | LIVE — the counted path can certify incomplete provenance, and the environment checks can misidentify a newline inside a value as a variable boundary. |
| Q3 · Residuals | SUGGESTION | — |

## Q1 · Landing and first-run requirements

**Before landing with the counted path enabled**

- **Check Git provenance before invoking the fleet.** Both entry points use `echo "git_head=$(git rev-parse HEAD)"`, whose exit status is that of `echo`. The status count also loses a failed `git status` behind `wc` and `echo`. Injecting exit 128 for each Git read in turn let both entry-point slices start all four rig stand-ins, pass the join and freeze, and reach the sampler boundary. One case froze an empty HEAD. The other presented the failed status read as zero lines. Earlier clean-tree admission is a separate read. Capture and validate the HEAD with its command status. Capture the status listing successfully before counting it, or use the existing clean-tree evidence as the tree-state record. Check the hash command's status directly as well. Cover the two Git failures on both entry points and require refusal before dispatch. `docs/drafts/cache_policy_v4/gate12_launch.diff:22`, `:35`, `:47`, `docs/drafts/cache_policy_v4/run4_launch.diff:22`, `:35`, `:47`, `gate12_launch.sh:246`, `/home/michael/run4_launch.sh:246`.
- **Preserve NUL boundaries in the startup and parked environment checks.** The parked probe sends the whole environment and turns NUL bytes into newlines. With a valid matching liveness row and fake `/proc`, the submitted join returned 0 and wrote PASS when there was no `CUPY_CACHE_DIR` variable and its spelling appeared on the second line of an unrelated value. It also passed with the expected path followed by a newline and additional characters inside the actual cache value. An unrelated non-UTF-8 value with the correct cache path caused a decoding error. The startup check uses the same newline conversion. Filter to the requested variable without losing NUL boundaries, check the read status and compare exactly one complete value. Preserve adjacent argv tokens when rechecking the nonce, following the existing liveness parser. Add missing, duplicate, multiline and correct-value controls. The fixtures establish the parser behavior. Whether such values are present on a rig is UNAVAILABLE. `docs/drafts/cache_policy_v4/launch_fleet_manual.diff:43`, `docs/drafts/cache_policy_v4/scripts/gate12_cache_join_gate.py:28`, `:93`, `:106`, `scripts/gate12_worker_liveness_gate.py:138`, `:423`.
- **Make publication agree with the verifier’s exit status.** The launcher refuses on nonzero and the v3 false success is resolved. A separate failure remains after rename. I injected an error on the directory `fsync` alone. The verifier returned 2 and left the final header and data row at the official TSV path, because cleanup only unlinks the temporary name. That filename is intended to signify successful publication, and the next check refuses it as already used. Handle the failure after publication or explicitly invalidate the output so it cannot serve as a success artifact, and test this boundary. The current interrupt tests inject before rename. `docs/drafts/cache_policy_v4/scripts/cache_records_verify.py:25`, `:93`, `:103`, `:126`, `:151`, `docs/drafts/cache_policy_v4/launch_fleet_manual.diff:118`, `:130`, `docs/drafts/cache_policy_v4/tests/test_cache_policy_v4.py:158`.
- **Make the test runnable in the intended landing layout.** The exact test file is a pre-apply draft harness. It locates the draft sources by its own location, reads the external Route A baseline, and reapplies the diffs to the repository launch files. After applying the patches to copies, a forward-only check rejected reapplication. Keep an explicit pre-apply fixture if that is the intended artifact. For a post-landing regression, point it at the landed scripts and supply an isolated Route A fixture. Retain a short executable exercise of each entry point through the joined evidence and freeze, with a passing manifest plus join and freeze failures that stop before the sampler. This can use local fixtures. `docs/drafts/cache_policy_v4/tests/test_cache_policy_v4.py:42`, `:434`, `:482`, `:491`, `docs/drafts/cache_policy_v4/gate12_launch.diff:54`, `docs/drafts/cache_policy_v4/run4_launch.diff:54`.

**Before the first counted run.** Use the clean committed harness and check the installed external Route A copy against it, including its digest and parameter-only diff. Keep the normal GPU and CT parity gates. The submitted cache-policy files sit outside the ten-file CT closure. If other changes touch governed files, the existing committed-HEAD deployment and 30/30 verification requirements apply. Arrange the timed sentinel capture in Q3 before dispatch. The run must produce the 24 verified rig rows and their joined frozen evidence before the sampler and coordinator, with Zeus recorded as local in the existing full-fleet liveness evidence. Measure the live timing and footprint on that first authorized run and review the retained evidence before accepting the result. `docs/RUN4_LAUNCH_CHECKLIST.md:376`, `docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:55`, `:59`, `:118`, `scripts/gate12_parity_gate.py:174`, `gate12_launch.sh:291`, `:331`, `docs/drafts/cache_policy_v4/gate12_launch.diff:58`, `docs/drafts/cache_policy_v4/run4_launch.diff:58`.

## Q2 · R3 checklist

| R3 item | Verified status |
|---|---|
| Literal parse and complete write | **Parser satisfied.** Exactly one candidate cache record, all ten unique keys, exact tag, dispatch host, GPU, policy, path, creation, count and environment, decimal prep time and digits for PID. A sole full-line ACK must match its GPU and PID. Slot count comes from the resolved fleet. The launcher handles a verifier or expectation-write failure as fleet failure. Fix the post-rename status/output discrepancy in Q1. `docs/drafts/cache_policy_v4/scripts/cache_records_verify.py:42`, `:58`, `:89`, `:129`, `docs/drafts/cache_policy_v4/launch_fleet_manual.diff:112`, `:124`. |
| Executable dispatch and entry-point evidence | **R3 dispatch exercise satisfied on the submitted drafts.** The launcher ran with inert processes whose actual environments were inspected. I reproduced clean per-run, rc 14, rc 15, the wrong-host record refusal, shared and Zeus controls, and the two entry-point slices refusing before the sentinel marker. The standalone passing join supplies liveness JSON from those PIDs. A separate Beta fixture ran both patched entry points through the real join and freeze with stubbed sentinel/liveness producers. Clean reached the sampler marker and verified nine hashes per bundle. Wrong parked PID stopped both with join rc 1. A blocked freeze directory stopped both with rc 97. This exercises wiring and the aborts. Real sentinel correlation remains run evidence. `docs/drafts/cache_policy_v4/tests/test_cache_policy_v4.py:353`, `:358`, `:385`, `:402`, `:434`, `:473`, `docs/drafts/cache_policy_v4/gate12_launch.diff:60`, `:65`, `docs/drafts/cache_policy_v4/run4_launch.diff:60`, `:65`. |
| Mandatory `per_run` | **Satisfied on both counted paths.** Each rejects any set value other than `per_run` and exports `per_run` before the launcher call. Both inherited `shared` cases refused before dispatch. The launcher-alone diagnostic control accepted `shared`. `docs/drafts/cache_policy_v4/gate12_launch.diff:11`, `:47`, `docs/drafts/cache_policy_v4/run4_launch.diff:11`, `:47`, `docs/drafts/cache_policy_v4/tests/test_cache_policy_v4.py:426`, `:464`. |
| Sentinel/liveness join and parked re-read | **Join keys and ordering satisfied.** The existing liveness gate binds the nonce, sentinel, PID, worker argv and release-wait evidence. Both entry points require its passing result before using the JSON. The new gate checks rig counts, nonce, endpoint, worker identity, GPU, PID and duplicates. The second environment read still needs the exact field handling in Q1. `scripts/gate12_worker_liveness_gate.py:340`, `:369`, `:423`, `:544`, `:564`, `gate12_launch.sh:331`, `docs/drafts/cache_policy_v4/scripts/gate12_cache_join_gate.py:57`, `:71`, `:93`, `docs/drafts/cache_policy_v4/gate12_launch.diff:50`, `docs/drafts/cache_policy_v4/run4_launch.diff:50`. |
| Provenance and freeze | **Placement satisfied.** Provenance precedes dispatch. Join and freeze precede sampler and coordinator. The freeze copies the verified/expected TSVs, dispatch logs, liveness, join and provenance. A failed copy or checksum command takes the same abort path as the join. Field completeness still has the Git defect in Q1. The freeze needs the additional timed sentinel capture in Q3 for the advertised measurement. `docs/drafts/cache_policy_v4/gate12_launch.diff:22`, `:47`, `:54`, `:65`, `:85`, `docs/drafts/cache_policy_v4/run4_launch.diff:22`, `:47`, `:54`, `:65`, `:85`, `gate12_launch.sh:376`. |

## Q3 · Residuals

- **L2 is equivalent for the current resolved fleet loop.** Each remote iteration appends one log and every local or remote iteration increments `DISPATCHED`. The verifier runs after the remote, local and truncated-dispatch refusal branches. A short loop cannot reach L2's reduced expectation. Retaining the fleet-derived count protects against future changes to that loop. `scripts/launch_fleet_manual.sh:188`, `:218`, `:272`, `:349`, `:416`, `:430`, `docs/drafts/cache_policy_v4/launch_fleet_manual.diff:97`, `:112`, `docs/drafts/cache_policy_v4/tests/test_cache_policy_v4.py:548`.
- **Full Gate 12 and Route A execution may wait for the first authorized run.** The local coverage needed for landing is the actual dispatch path, the counted-path refusals and the join/freeze boundary. Clean-tree, GPU, parity, real sentinel/liveness, coordinator and release remain part of the operational run. `gate12_launch.sh:133`, `:164`, `:191`, `:291`, `:331`, `:376`, `:398`.
- **The timing and footprint plan needs one correction before dispatch.** The v4 freeze omits the timed sentinel records, and the liveness JSON stores per-worker results with counts and PID. The worker log supplies the sentinel UTC timestamp. Freeze the nonce- and PID-matched sentinel line for each rig slot, with its timestamp, before the logs can be reused. Subtract the same CT's epoch `t_prep` from that timestamp and report it alongside the stagger and the time from first parked worker to release. For footprint, read the joined paths from the TSV so an allowed cache-root override is measured correctly. Record the per-CT retained directory count and say whether the reported sizes are apparent or allocated bytes. The measured values belong to the first run's report. `docs/drafts/cache_policy_v4/README.md:45`, `docs/drafts/cache_policy_v4/gate12_launch.diff:65`, `docs/drafts/cache_policy_v4/run4_launch.diff:65`, `scripts/gate12_worker_liveness_gate.py:242`, `:637`, `miner/range_miner_worker.py:2157`, `:3004`, `scripts/launch_fleet_manual.sh:150`, `docs/drafts/cache_policy_v4/launch_fleet_manual.diff:22`, `:41`.
- **Zeus remains an appropriate local control.** The evidence and policy concern the ROCm rig workers. The local CUDA path supplies its own cache and still participates in sentinel and liveness checks. The stand-in control saw the local path, and the join has a separate local evidence list. `docs/LEADS.md:590`, `scripts/launch_fleet_manual.sh:276`, `:286`, `scripts/gate12_sentinel_gate.py:133`, `docs/drafts/cache_policy_v4/scripts/gate12_cache_join_gate.py:59`, `docs/drafts/cache_policy_v4/tests/test_cache_policy_v4.py:377`.

## Checks and limits

Activated `~/venvs/torch`. Ran the submitted suite and mutation mode on a copy outside the worktree. The initial relocation under `/tmp` exposed a fixture issue. The SSH stub blindly rewrites `/tmp/` inside the real fixture path and makes the remote `cd` fail. After shielding that prefix in the copied stub and pointing Route A at a copied baseline, I reproduced **65/65 checks** and **14/15 mutants detected**, with L2 the only survivor. The production draft bytes were unchanged in those checks. Ran the additional stubbed entry-point slices, Git errors, environment byte cases, and post-rename failure described above. `docs/drafts/cache_policy_v4/tests/test_cache_policy_v4.py:47`, `:274`, `:297`, `:491`, `:559`.

Read the checked-in sentinel and liveness producers, release ordering and CT parity scope. Full end-to-end Gate 12 and Route A, rig measurements and held-out pool hit rate are **UNAVAILABLE**. The reported cold/warm observations concern launch reliability under the recorded test conditions. Michael decides landing and counted-run acceptance. `docs/LEADS.md:590`, `gate12_launch.sh:291`, `:331`, `:405`, `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:317`.
