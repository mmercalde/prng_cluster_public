## Prior record

**Searches run.** Searched `docs/` with `rg` for `X-PWC-CACHE`, `pwc_cache_reset`, persistent worker coordinator, ready gate, stale workers, reconnects, warm caches and SSH failures. The search covered the exchanges, catalog, backlog, handoffs, `LEADS.md`, `TB_RULING_*`, `PROPOSAL_*`, `CHAPTER_*_AUDIT.md` and `TODO_CONSOLIDATED.md`. Read the R1 exchange and its carried records, the R2 request and logs, the exact draft, and the startup and dispatch paths. Checked `git show`, `git hash-object` and `sha256sum` against main `1e6e675`.

- **ALREADY RECORDED.** R1 documents the TCP admission gap. S156, VIR-1, the S185 resolved execution set and the historical fault register cover cleanup, proof of completion, roster scope and orphan reconnects. R2 carries these records. `docs/exchange/from_beta/X-PWC-CACHE-RESET_R1_BETA_REVIEW.md:85-95`, `docs/PROPOSAL_PWC_LIFECYCLE_FIX_S156_v2_0.md:67-77`, `docs/VERIFICATION_INTEGRITY_STANDARD.md:16-35`, `docs/CLAUDE_CODE_INSTRUCTIONS_RESOLVED_EXECUTION_SET.md:38-57`, `5605ed3:docs/REGISTER_GPU_FAULT_PRIOR_WORK.md:1621-1625`.
- **ALREADY RECORDED.** The handoffs cover the fixed cache paths, the owner's reboot/rejoin observation and the pending landing decision. The run-scope note bounds the warm-cache evidence to the miner. `docs/TFM_SESSION_HANDOFF_20261118.md:56-60`, `docs/TFM_SESSION_HANDOFF_20261118_ADDENDUM.md:36-41`, `:65-71`, `docs/drafts/NOTE_RUN_SCOPE_DECISION_20261118.md:31-35`.
- **ALREADY RECORDED.** The fleet-state correction records the effective one-worker gate and Michael's August 4 decision. See Finding 1.
- **NEW.** Michael's R2 scope, Alpha's R2 before-and-after logs and Beta's R2 disposition. **CHANGES A RULING.** None.

**Outside the search.** Live rig access, Proxmox, live run state, ser8, the private repository and private conversations.

# X-PWC-CACHE-RESET · R2 · Team Beta review

**Date and class.** 2026-11-19, America/Los_Angeles. Before-commit coordinator review.

**Base read.** Main `1e6e675636559a4972d7bb5d69f54185b4142f6d`. Worktree HEAD `3579088b8e9c2d902252d626baae15bf0647907f`. The coordinator, TCP transport and worker service match the named base. Both draft hashes match the request.

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
?? docs/TFM_SESSION_HANDOFF_20261118_ADDENDUM.md
?? docs/drafts/
?? docs/exchange/from_beta/X-CACHE-POLICY-D6_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-CACHE-POLICY_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-CACHE-POLICY_R2_BETA_REVIEW.md
?? docs/exchange/from_beta/X-CACHE-POLICY_R3_BETA_REVIEW.md
?? docs/exchange/from_beta/X-GCVM-LOCATED_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-GCVM-LOCATED_R2_BETA_REVIEW.md
?? docs/exchange/from_beta/X-GCVM-LOCATED_R3_BETA_REVIEW.md
?? docs/exchange/from_beta/X-LEDGER-RELEASE_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-MINER-NO-TORCH_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-PWC-CACHE-RESET_R1_BETA_REVIEW.md
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
?? docs/exchange/to_beta/X-LEDGER-RELEASE_R1_ALPHA_REVIEW_REQUEST_hand_release_run_I.md
?? docs/exchange/to_beta/X-MINER-NO-TORCH_R1_ALPHA_REVIEW_REQUEST_cleanup_torch_flag.md
?? docs/exchange/to_beta/X-PWC-CACHE-RESET_R1_ALPHA_REVIEW_REQUEST_pwc_cache_reset_v1.md
?? docs/exchange/to_beta/X-PWC-CACHE-RESET_R2_ALPHA_REVIEW_REQUEST_real_world_scope.md
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
| Q1 — Landable as submitted? **Yes** | NO OBJECTION | — |

## Finding 1 — Governing record omitted from R2

**TO-DO.** Cite the August 4 correction and owner ruling on PWC readiness. The manifest lists `min_workers=24`, but the Window Optimizer's call omits the argument. The shim supplies `1` to the coordinator, so one ready worker satisfies the count at startup and before dispatch. Michael's ruling leaves the full-fleet utilization check retired on this backup path. R1's 17/16 is an observed count against launches in a fixture. The effective minimum is one. `docs/FLEET_STATE_REQUIREMENTS_v1.md:146-195`, `agent_manifests/window_optimizer.json:262-269`, `window_optimizer_integration_final.py:1968-1986`, `persistent_worker_coordinator.py:1649`, `:1685`, `:851-869`, `:894-908`, `:1249-1264`.

## Q1 — Yes, landable as submitted

**TO-DO (a) — Common cause.** A short rig-side SSH service outage after the GPU probe could strand both commands while the rig and its PWC TCP connection remain usable. For example, a PWC coordinator crashes, an old rig worker survives and retries its connection, and the replacement coordinator binds at the worker's existing dial address. If that rig stops accepting SSH sessions during both the TCP reap and the reset, neither command reaches the rig. The reset refusal skips new launches. The old service can still reconnect, report ready and request a job. This scenario requires the probe to succeed first because a failed probe already skips the rig on main. The historical orphan incident establishes survival after coordinator loss. The selective SSH outage in this sequence is an inference. `5605ed3:docs/REGISTER_GPU_FAULT_PRIOR_WORK.md:1621-1625`, `persistent_worker_coordinator.py:513-529`, `:405-412`, `:691-726`, `:748-757`, `docs/drafts/pwc_cache_reset_v1/persistent_worker_coordinator.diff:51-64`, `:69-75`, `persistent/pwc_worker_service.py:169-213`, `persistent/pwc_transport_tcp.py:393-451`, `:514-550`.

**TO-DO (b) — Likelihood here.** Zero current exposure while PWC is idle. If the operator invokes the backup, I expect the combined event to be uncommon. It requires an earlier PWC worker to survive and retry while the coordinator is restarted, plus loss of SSH at the two command attempts while TCP is usable. The old S151 SSH-rejection incident was a different mechanism and its SSH configuration was corrected. The record has no fleet rate for this combination. R1 observed assignment of a fixture job after injected failures. Fleet occurrence, GPU faults and wrong results from this sequence are **UNAVAILABLE**. I withdraw the R1 blocking classification under Michael's R2 scope. `docs/exchange/to_beta/X-PWC-CACHE-RESET_R2_ALPHA_REVIEW_REQUEST_real_world_scope.md:23-34`, `docs/ROOT_CAUSE_ANALYSIS_RRIG6600C_S151.md:9-15`, `:114-128`, `docs/exchange/from_beta/X-PWC-CACHE-RESET_R1_BETA_REVIEW.md:89-95`.

**TO-DO (c) — Safer than main for cache freshness.** Main uses the fixed rig paths and reuses existing contents when present. A first launch after reboot can already be cold. The patch empties and checks those directories before it starts new rig workers. A failed reset withholds fresh launches. In TCP the transport can still count and serve an old worker from that rig on either version. The added availability cost is that a reset-only failure can leave the run short a rig even if SSH would have recovered in time for main's launch commands. Two healthy rigs can still satisfy the effective one-worker minimum. This is a startup and capacity trade-off. L-7's 8/8 cold attempts clean and 4/4 warm attempts faulting were miner phase-4 observations on two rigs. The PWC completion and fault-rate effect is **UNAVAILABLE**. `persistent_worker_coordinator.py:593-611`, `:728-767`, `:851-902`, `docs/drafts/pwc_cache_reset_v1/persistent_worker_coordinator.diff:16-24`, `:31-46`, `:51-64`, `:69-75`, `persistent/pwc_transport_tcp.py:393-451`, `:537-550`, `docs/TFM_SESSION_HANDOFF_20261118.md:56-60`, `5605ed3:docs/LEADS.md:582-592`, `docs/drafts/NOTE_RUN_SCOPE_DECISION_20261118.md:31-35`.

**TO-DO — Carry the known TCP limit forward.** The draft README's refusal description applies to launch attempts and the online wait target. The TCP listener can still initialize, count and give jobs to reconnecting workers. Keep the R1 follow-up for matching admission to successful cache preparation and the current launch, with a test through the real readiness and assignment path. When PWC is next evaluated, a same-boot warm-versus-reset comparison with worker identities recorded would measure the effect on PWC fault and completion rates. `docs/drafts/pwc_cache_reset_v1/README.md:10-14`, `docs/drafts/pwc_cache_reset_v1/persistent_worker_coordinator.diff:69-75`, `persistent_worker_coordinator.py:836-846`, `persistent/pwc_transport_tcp.py:393-451`, `:537-550`, `docs/drafts/pwc_cache_reset_v1/tests/test_pwc_cache_reset.py:140-165`.

**TO-DO — Evidence limits.** I activated `~/venvs/torch`, applied the exact diff with `--fuzz=0` to the named base in a temporary directory, and ran the submitted CPU-only suite. **21/21 passed.** I read Alpha's six existing-suite comparisons. The three baseline failures recur after the patch. Phase 4 reports 62/63 with the new test untracked and 63/63 with it set aside. I did not rerun those six suites or the R1 reconnect fixture. Real SSH, rig execution, a PWC warm/cold comparison and committed-tree Gate 22 are **UNAVAILABLE** here. `docs/drafts/pwc_cache_reset_v1/tests/test_pwc_cache_reset.py:2-7`, `:140-165`, `logs/pwc_cache_v1/r2/before/SUMMARY.txt:1-8`, `logs/pwc_cache_v1/r2/after/SUMMARY.txt:1-10`, `logs/pwc_cache_v1/r2/after/test_s172_phase4_coordinator.log:34`.

Michael decides whether to land.
