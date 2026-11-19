## Prior record

**Searches run.** Read the request, draft, tests and both coordinator start paths. Searched `docs/`, including `TB_RULING_*`, `PROPOSAL_*`, `CHAPTER_*_AUDIT.md`, `TODO_CONSOLIDATED.md`, and the exchanges, for PWC cache reset, stale-worker reap, reconnect, admission and Gate 22. Read the historical fault register at `5605ed3` because it is absent in this checkout. Verified commit `9034e18` and compared the coordinator blob at both named bases and the worktree HEAD.

- **ALREADY RECORDED.** L-7 records 8/8 cold attempts clean and 4/4 warm attempts faulting on the two affected rigs at miner phase 4. Commit `9034e18` makes per-run caches the range-miner default. The scope note records the fixed PWC paths and the separate evidence limits. `5605ed3:docs/LEADS.md:582-592`, `docs/drafts/NOTE_RUN_SCOPE_DECISION_20261118.md:31-35`.
- **ALREADY RECORDED.** The session handoff records Michael's draft authorization and the separate landing decision. S156 sets pre-spawn cleanup expectations. VIR-1 requires evidence of completion and records a prior cleanup command that matched its own shell. S185 defines frozen fleet membership and describes miner-specific registration enforcement. The fault register records reconnects and orphaned TCP workers after coordinator loss. See Finding 1.
- **NEW.** The submitted PWC reset and the reproduced TCP refusal bypass below. **CHANGES A RULING.** None.

**Outside the search.** Private chat and repository, ser8, rigs, Proxmox, live state and deployment.

# X-PWC-CACHE-RESET · R1 · Team Beta review

**Date and class.** 2026-11-18, America/Los_Angeles. Before-commit coordinator review.

**Base read.** Main `5605ed35298ac50cfcb06423cf5090c029a85f46`, earlier base `fe763fe0e625bffc15fdc02f997212c0d703e1d6`, worktree HEAD `3579088b8e9c2d902252d626baae15bf0647907f`. The coordinator blob is `36bacb708177fe6b6469bc327f88dcaf075b30ef` at all three. The worktree coordinator and the three submitted artifact hashes match the request. The diff applied to the named main base with zero fuzz in a temporary directory.

**First-read `git status --porcelain`**

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
?? docs/RIG_POWER_CONTROL_TUYA_REFERENCE.md
?? docs/SESSION_CHANGELOG_20261115_RIG_GPU_FAULT.md
?? docs/SESSION_CHANGELOG_20261116_RIG_FAULT_RECORDER.md
?? docs/TFM_SESSION_HANDOFF_20261118.md
?? docs/drafts/
?? docs/exchange/from_beta/X-CACHE-POLICY-D6_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-CACHE-POLICY_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-CACHE-POLICY_R2_BETA_REVIEW.md
?? docs/exchange/from_beta/X-CACHE-POLICY_R3_BETA_REVIEW.md
?? docs/exchange/from_beta/X-GCVM-LOCATED_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-GCVM-LOCATED_R2_BETA_REVIEW.md
?? docs/exchange/from_beta/X-GCVM-LOCATED_R3_BETA_REVIEW.md
?? docs/exchange/from_beta/X-MINER-NO-TORCH_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-RIG-GPU-FAULT_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-RIG-GPU-FAULT_R2_BETA_REVIEW.md
?? docs/exchange/from_beta/X-RIG-GPU-FAULT_R3_BETA_REVIEW.md
?? docs/exchange/from_beta/X-RIG-GPU-PROBE_R1_BETA_REVIEW.md
?? docs/exchange/to_beta/X-CACHE-POLICY-D6_R1_ALPHA_REVIEW_REQUEST_cache_policy_v7.md
?? docs/exchange/to_beta/X-CACHE-POLICY_R1_ALPHA_REVIEW_REQUEST_cache_policy_v4.md
?? docs/exchange/to_beta/X-CACHE-POLICY_R2_ALPHA_REPLY.md
?? docs/exchange/to_beta/X-CACHE-POLICY_R3_ALPHA_REPLY.md
?? docs/exchange/to_beta/X-CUPY-CACHE-RESET_R1_ALPHA_REVIEW_REQUEST_launcher_cache_reset.md
?? docs/exchange/to_beta/X-GCVM-LOCATED_R1_ALPHA_REVIEW_REQUEST_fault_address_reading.md
?? docs/exchange/to_beta/X-GCVM-LOCATED_R2_ALPHA_REPLY.md
?? docs/exchange/to_beta/X-GCVM-LOCATED_R3_ALPHA_REPLY.md
?? docs/exchange/to_beta/X-MINER-NO-TORCH_R1_ALPHA_REVIEW_REQUEST_cleanup_torch_flag.md
?? docs/exchange/to_beta/X-PWC-CACHE-RESET_R1_ALPHA_REVIEW_REQUEST_pwc_cache_reset_v1.md
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

## Positions

| Question | Position | Severity |
|---|---|---|
| Q1 — Landability of the diff and test | OBJECTION | LIVE — A TCP worker from a rig whose fresh-cache check failed can reconnect, count as ready and take a job. |

## Finding 1 — Governing records missing from the request

**TO-DO.** Cite the session handoff for Michael's authorization to draft and his separate decision on landing. Include the S156 lifecycle proposal and the verification-integrity standard for the cleanup dependency. Include the S185 frozen fleet rule for the existing roster and the historical register for the observed persistence of orphan workers. The S185 registration gate is miner-specific. These are recorded constraints and observations. [Session handoff](../../TFM_SESSION_HANDOFF_20261118.md#decisions-by-michael-this-session), `docs/PROPOSAL_PWC_LIFECYCLE_FIX_S156_v2_0.md:67-77`, `docs/VERIFICATION_INTEGRITY_STANDARD.md:16-35`, `docs/CLAUDE_CODE_INSTRUCTIONS_RESOLVED_EXECUTION_SET.md:38-57`, `5605ed3:docs/REGISTER_GPU_FAULT_PRIOR_WORK.md:1621-1625`.

## Q1 — NOT LANDABLE as submitted

**BLOCKS. TCP cache refusal ends at the launch loop.** The TCP listener starts before the rig reap, and the reap call ignores a nonzero SSH result. A reset refusal skips new launch scripts and lowers the expected launch count. The transport still accepts any connecting worker ID, sends init, counts its ready message and serves its job requests. A PWC worker retries connections after transport loss, and the project recorded 26 orphan workers surviving coordinator loss. For a fleet whose workers can reach the PWC listener, a brief SSH-session failure can leave the old service alive and make the reset fail while the service can still reach the TCP listener. The old process can then receive work without verified fresh-cache preparation. The test demonstrates job admission in that condition. Effects on GPU faults and sieve results remain unmeasured. `persistent_worker_coordinator.py:513-529`, `:712-726`, `:836-846`, `:851-902`, `docs/drafts/pwc_cache_reset_v1/persistent_worker_coordinator.diff:51-64`, `:69-75`, `persistent/pwc_transport_tcp.py:248-281`, `:393-451`, `:514-550`, `persistent/pwc_worker_service.py:169-213`, `:297-306`, `5605ed3:docs/REGISTER_GPU_FAULT_PRIOR_WORK.md:1621-1625`.

**BLOCKS reproducer.** In a temporary copy of `5605ed3` with the submitted diff applied, run the real TCP launch method, reset-result parser, transport handler, init broadcast and barriers. Stub the live GPU probe to eight per rig and simulate SSH return code 255 for the reap and reset on `.156`. Feed a simulated reconnect from `192_168_3_156_gpu0` through the real TCP handler, then give it a fixture job to request. The other two rigs supply 16 simulated fresh peers. The test uses in-memory messages and subprocess stand-ins. With the stale peer absent, both barriers report 16/16. With the stale peer present, there are still zero launch-script calls for `.156`, both barriers report 17/16, and the peer receives `hello_ack`, `init` and `job_assign` for `fixture-job`. The runnable local fixture and output are at `/tmp/tfm-pwc-beta.IOSVM6QY/beta_tcp_refusal.py` and `/tmp/tfm-pwc-beta.IOSVM6QY/beta_tcp_refusal.log`. The submitted TCP test substitutes the init and ready methods and checks the expected online count, which is 16 in both cases. `docs/drafts/pwc_cache_reset_v1/tests/test_pwc_cache_reset.py:140-165`.

**BLOCKS closure.** Enforce successful cache preparation and current-launch identity at TCP admission so an old peer using the same worker ID cannot qualify. Cover refused and late peers through the real handler, readiness and job assignment. If keeping the fixed cache paths, require evidence that old processes using those paths have stopped before declaring the reset complete. `persistent_worker_coordinator.py:712-736`, `docs/drafts/pwc_cache_reset_v1/persistent_worker_coordinator.diff:16-24`, `:69-75`, `persistent/pwc_transport_tcp.py:393-426`, `:437-451`.

**TO-DO. SSH-PWC and test coverage.** The refusal creates quarantined handles, which the respawn guard refuses. The test labeled W7 checks target spelling and the number of resets. Record spawn events alongside resets if keeping W7's ordering claim. The successful-path source places the reset before spawn. `docs/drafts/pwc_cache_reset_v1/persistent_worker_coordinator.diff:31-46`, `persistent_worker_coordinator.py:923-943`, `docs/drafts/pwc_cache_reset_v1/tests/test_pwc_cache_reset.py:167-187`.

**TO-DO. Evidence limits and recorded statuses.** I activated `~/venvs/torch`, applied the exact diff and ran the submitted test from its landed path in the temporary tree. **21/21 passed.** I also ran the TCP reproducer above. Alpha's six existing-suite comparisons and five mutants are reported results. I read those summaries. Gate 22's dirty-tree status is already recorded. The PWC dial-address caveat on VM101 is also recorded. The reproducer uses in-memory messages. Real SSH, fleet or GPU behavior, clean committed-tree Gate 22 closure and a PWC warm/cold comparison are **UNAVAILABLE** in this review. The owner-reported reboot/rejoin observation motivates the change. A same-boot PWC comparison with warm versus freshly prepared caches and verified worker identities would test whether the reset changes fault or completion rates on PWC. `docs/exchange/to_beta/X-PWC-CACHE-RESET_R1_ALPHA_REVIEW_REQUEST_pwc_cache_reset_v1.md:43-59`, `docs/TB_RULING_FIELD6_IMPLEMENTATION.md:112-120`, `persistent_worker_coordinator.py:748-757`, `docs/CLAUDE_CODE_REPORT_PWC_BACKUP_STAGE_A.md:308-311`, `docs/TFM_SESSION_HANDOFF_20261118.md:56-60`.
